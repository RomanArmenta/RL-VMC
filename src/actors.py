from __future__ import annotations

"""Actor architectures for the shared single-electron RL proposal policy.

The actor architecture is deliberately separated from the proposal *mode*.
The proposal mode (scale_only / residual_mala / full_gaussian) determines how
actor outputs are converted into a Gaussian Metropolis-Hastings proposal.
This module determines how the state is represented and how the neural actor
maps that state to bounded action components.
"""

import math
from typing import Literal

import jax
import jax.numpy as jnp

from src.physics import SystemConfig, spin_labels

Array = jax.Array
ActorArchitecture = Literal["flat_mlp", "multibranch_deepset"]

from flax import linen as nn
import jax.numpy as jnp
import jax
from functools import partial


class DirectionalActor(nn.Module):
    n_elec: int
    n_nuclei: int
    hidden_dim: int = 64
    actor_amplitude: float = 0.05

    def setup(self):
        self.dense1 = nn.Dense(features=self.hidden_dim)
        self.dense2 = nn.Dense(features=self.hidden_dim)
        self.out_w_nuclei = nn.Dense(features=self.n_nuclei, kernel_init=nn.initializers.zeros)
        self.out_w_elec = nn.Dense(features=self.n_elec - 1, kernel_init=nn.initializers.zeros)
        self.out_sigma = nn.Dense(features=3, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros)

    def __call__(self, walkers, elec_idx, r_nuclei, exploration_mode=True):
        r_i = walkers[:, elec_idx, :][:, None, :]
        
        vec_iI = r_i - r_nuclei[None, :, :] + 1e-8
        d_iI = jnp.linalg.norm(vec_iI, axis=-1, keepdims=True)
        u_iI = vec_iI / (d_iI)
        
        all_indices = jnp.arange(self.n_elec)
        shifted_indices = jnp.roll(all_indices, -elec_idx)
        others_idx = shifted_indices[1:]
        
        r_j = walkers[:, others_idx, :]
        vec_ij = r_i - r_j + 1e-8
        d_ij = jnp.linalg.norm(vec_ij, axis=-1, keepdims=True)
        u_ij = vec_ij / (d_ij)
        
        features = jnp.concatenate([jnp.exp(-d_iI).reshape(-1, self.n_nuclei), 
                                    jnp.exp(-d_ij).reshape(-1, self.n_elec - 1)], axis=-1)
        
        x = nn.relu(self.dense1(features))
        x = nn.relu(self.dense2(x))
        
        w_nuclei = self.out_w_nuclei(x)
        w_elec = self.out_w_elec(x)
        
        drift_nuclei = jnp.sum(w_nuclei[..., None] * u_iI, axis=1)
        drift_elec = jnp.sum(w_elec[..., None] * u_ij, axis=1)
        
        explore_factor = jnp.where(exploration_mode, 1.0, 0.0)
        drift_res = self.actor_amplitude * nn.tanh(drift_nuclei + drift_elec) * explore_factor

        sigma_raw = self.out_sigma(x)
        # Apagamos la alteración de la varianza si no estamos explorando
        sigma_scale = nn.tanh(sigma_raw) * explore_factor
        
        return drift_res, sigma_scale

class DirectionalActorCovariant(nn.Module):
    n_elec: int
    n_nuclei: int
    hidden_dim: int = 64
    actor_amplitude: float = 0.05

    def setup(self):
        self.dense1 = nn.Dense(features=self.hidden_dim)
        self.dense2 = nn.Dense(features=self.hidden_dim)
        self.out_w_nuclei = nn.Dense(features=self.n_nuclei, kernel_init=nn.initializers.zeros)
        self.out_w_elec = nn.Dense(features=self.n_elec - 1, kernel_init=nn.initializers.zeros)
        
        # Pesos para las matrices direccionales (tensor-based) y base isotrópica
        self.out_w_cov_nuclei = nn.Dense(features=self.n_nuclei, kernel_init=nn.initializers.zeros)
        self.out_sigma_iso = nn.Dense(features=1, kernel_init=nn.initializers.zeros)

    def __call__(self, walkers, elec_idx, r_nuclei, exploration_mode=True):
        r_i = walkers[:, elec_idx, :][:, None, :]
        
        vec_iI = r_i - r_nuclei[None, :, :] + 1e-8
        d_iI = jnp.linalg.norm(vec_iI, axis=-1, keepdims=True)
        u_iI = vec_iI / d_iI
        
        all_indices = jnp.arange(self.n_elec)
        shifted_indices = jnp.roll(all_indices, -elec_idx)
        others_idx = shifted_indices[1:]
        
        r_j = walkers[:, others_idx, :]
        vec_ij = r_i - r_j + 1e-8
        d_ij = jnp.linalg.norm(vec_ij, axis=-1, keepdims=True)
        u_ij = vec_ij / d_ij
        
        features = jnp.concatenate([jnp.exp(-d_iI).reshape(-1, self.n_nuclei), 
                                    jnp.exp(-d_ij).reshape(-1, self.n_elec - 1)], axis=-1)
        
        x = nn.relu(self.dense1(features))
        x = nn.relu(self.dense2(x))
        
        explore_factor = jnp.where(exploration_mode, 1.0, 0.0)
        
        # 1. Cálculo del Drift
        w_nuclei = self.out_w_nuclei(x)
        w_elec = self.out_w_elec(x)
        drift_nuclei = jnp.sum(w_nuclei[..., None] * u_iI, axis=1)
        drift_elec = jnp.sum(w_elec[..., None] * u_ij, axis=1)
        drift_res = self.actor_amplitude * nn.tanh(drift_nuclei + drift_elec) * explore_factor

        # 2. Cálculo de la Matriz de Covarianza (3x3)
        # Usamos Softplus para garantizar que los pesos aporten varianza positiva
        w_cov = jax.nn.softplus(self.out_w_cov_nuclei(x)) * explore_factor
        
        # Producto exterior: u @ u^T -> Shape (batch, n_nuclei, 3, 3)
        u_outer = jnp.einsum('bni,bnj->bnij', u_iI, u_iI)
        
        # Tensor direccional ponderado por la red -> Shape (batch, 3, 3)
        sigma_dir = jnp.einsum('bn,bnij->bij', w_cov, u_outer)
        
        # Base isotrópica (Identidad) para evitar matrices singulares
        s_iso = jax.nn.softplus(self.out_sigma_iso(x)) * explore_factor
        ident = jnp.eye(3).reshape(1, 3, 3)
        sigma_iso = s_iso[..., None] * ident
        
        # Matriz final simétrica y covariante
        Sigma = sigma_iso + sigma_dir + (1e-5 * ident)
        
        # Devolvemos la matriz de Cholesky (L) para generar el ruido
        # L tiene shape (batch, 3, 3)
        sigma_cholesky = jnp.linalg.cholesky(Sigma)
        
        return drift_res, sigma_cholesky

class DirectionalActorIsotropic(nn.Module):
    n_elec: int
    n_nuclei: int
    hidden_dim: int = 64
    actor_amplitude: float = 0.05

    def setup(self):
        self.dense1 = nn.Dense(features=self.hidden_dim)
        self.dense2 = nn.Dense(features=self.hidden_dim)
        self.out_w_nuclei = nn.Dense(features=self.n_nuclei, kernel_init=nn.initializers.zeros)
        self.out_w_elec = nn.Dense(features=self.n_elec - 1, kernel_init=nn.initializers.zeros)
        
        # EL CAMBIO CLAVE: feature=1 en lugar de 3. 
        # La red ahora predice un solo factor de escala para mantener la isotropía espacial.
        self.out_sigma = nn.Dense(features=1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros)

    def __call__(self, walkers, elec_idx, r_nuclei, exploration_mode=True):
        r_i = walkers[:, elec_idx, :][:, None, :]
        
        vec_iI = r_i - r_nuclei[None, :, :] + 1e-8
        d_iI = jnp.linalg.norm(vec_iI, axis=-1, keepdims=True)
        u_iI = vec_iI / d_iI
        
        all_indices = jnp.arange(self.n_elec)
        shifted_indices = jnp.roll(all_indices, -elec_idx)
        others_idx = shifted_indices[1:]
        
        r_j = walkers[:, others_idx, :]
        vec_ij = r_i - r_j + 1e-8
        d_ij = jnp.linalg.norm(vec_ij, axis=-1, keepdims=True)
        u_ij = vec_ij / d_ij
        
        features = jnp.concatenate([jnp.exp(-d_iI).reshape(-1, self.n_nuclei), 
                                    jnp.exp(-d_ij).reshape(-1, self.n_elec - 1)], axis=-1)
        
        x = nn.relu(self.dense1(features))
        x = nn.relu(self.dense2(x))
        
        w_nuclei = self.out_w_nuclei(x)
        w_elec = self.out_w_elec(x)
        
        drift_nuclei = jnp.sum(w_nuclei[..., None] * u_iI, axis=1)
        drift_elec = jnp.sum(w_elec[..., None] * u_ij, axis=1)
        
        # Apagado total del drift y varianza en modo evaluación
        explore_factor = jnp.where(exploration_mode, 1.0, 0.0)
        drift_res = self.actor_amplitude * nn.tanh(drift_nuclei + drift_elec) * explore_factor

        # Se predice un solo valor y se replica 3 veces para los ejes (x, y, z)
        sigma_raw = self.out_sigma(x) # Shape: (batch, 1)
        sigma_scalar = nn.tanh(sigma_raw) * explore_factor
        sigma_scale = jnp.tile(sigma_scalar, (1, 3)) # Shape: (batch, 3)
        
        return drift_res, sigma_scale

class DirectionalActorForceAligned(nn.Module):
    n_elec: int
    n_nuclei: int
    hidden_dim: int = 64
    actor_amplitude: float = 0.05

    def setup(self):
        # Tronco de características principales
        self.dense1 = nn.Dense(features=self.hidden_dim)
        self.dense2 = nn.Dense(features=self.hidden_dim)
        
        # Cabezas para predecir los pesos del vector de fuerza \omega(r)
        self.out_w_nuclei = nn.Dense(features=self.n_nuclei, kernel_init=nn.initializers.zeros)
        self.out_w_elec = nn.Dense(features=self.n_elec - 1, kernel_init=nn.initializers.zeros)
        
        # Cabezas escalares para las varianzas principales
        self.out_sigma_para = nn.Dense(features=1, kernel_init=nn.initializers.zeros)
        self.out_sigma_perp = nn.Dense(features=1, kernel_init=nn.initializers.zeros)

    def __call__(self, walkers, elec_idx, r_nuclei, exploration_mode=True):
        # --- 1. Geometría y Vectores Unitarios ---
        r_i = walkers[:, elec_idx, :][:, None, :]
        
        vec_iI = r_i - r_nuclei[None, :, :] + 1e-8
        d_iI = jnp.linalg.norm(vec_iI, axis=-1, keepdims=True)
        u_iI = vec_iI / d_iI
        
        all_indices = jnp.arange(self.n_elec)
        shifted_indices = jnp.roll(all_indices, -elec_idx)
        others_idx = shifted_indices[1:]
        
        r_j = walkers[:, others_idx, :]
        vec_ij = r_i - r_j + 1e-8
        d_ij = jnp.linalg.norm(vec_ij, axis=-1, keepdims=True)
        u_ij = vec_ij / d_ij
        
        # Base asintótica e^{-r}
        features = jnp.concatenate([jnp.exp(-d_iI).reshape(-1, self.n_nuclei), 
                                    jnp.exp(-d_ij).reshape(-1, self.n_elec - 1)], axis=-1)
        
        # --- 2. Procesamiento de la Red Neuronal ---
        x = nn.relu(self.dense1(features))
        x = nn.relu(self.dense2(x))
        
        explore_factor = jnp.where(exploration_mode, 1.0, 0.0)
        
        # --- 3. Construcción del Vector de Fuerza ---
        w_nuclei = self.out_w_nuclei(x)
        w_elec = self.out_w_elec(x)
        
        f_nuclei = jnp.sum(w_nuclei[..., None] * u_iI, axis=1)
        f_elec = jnp.sum(w_elec[..., None] * u_ij, axis=1)
        f_total = f_nuclei + f_elec
        
        sq_norm = jnp.sum(f_total**2, axis=-1, keepdims=True)
        norm_f = jnp.sqrt(sq_norm + 1e-8)
        f_hat = f_total / norm_f
        
        drift_res = self.actor_amplitude * nn.tanh(f_total) * explore_factor

        # --- 4. Construcción Segura de la Matriz de Covarianza ---
        raw_para = self.out_sigma_para(x)
        raw_perp = self.out_sigma_perp(x)
        
        # CLIPPING ESTRICTO: Previene la explosión a infinito del RL
        safe_para = jnp.clip(raw_para, -5.0, 1.0)
        safe_perp = jnp.clip(raw_perp, -5.0, 1.0)
        
        # ESCALAMIENTO: jnp.exp asegura valores positivos acotados
        sigma_para = jnp.exp(safe_para)
        sigma_perp = jnp.exp(safe_perp)
        
        # MODO EVALUACIÓN: Si está apagado, forzamos a 1.0 (Matriz Identidad para MALA clásico)
        sigma_para = jnp.where(exploration_mode, sigma_para, 1.0)
        sigma_perp = jnp.where(exploration_mode, sigma_perp, 1.0)
        
        sigma_para_sq = sigma_para ** 2
        sigma_perp_sq = sigma_perp ** 2
        
        ident = jnp.eye(3).reshape(1, 3, 3)
        f_outer = jnp.einsum('bi,bj->bij', f_hat, f_hat)
        
        term1 = sigma_perp_sq[..., None] * ident
        term2 = (sigma_para_sq - sigma_perp_sq)[..., None] * f_outer
        
        Sigma = term1 + term2 
        
        # Estabilizador numérico
        Sigma = Sigma + (1e-5 * ident)
        
        sigma_cholesky = jnp.linalg.cholesky(Sigma)

        # --- NUEVO: Construcción del diccionario de diagnóstico ---
        actor_stats = {
            "mean_sigma_para": jnp.mean(sigma_para),
            "mean_sigma_perp": jnp.mean(sigma_perp),
            "max_w_nuc": jnp.max(jnp.abs(w_nuclei)),
            "max_w_elec": jnp.max(jnp.abs(w_elec)),
            "mean_drift_act": jnp.mean(jnp.linalg.norm(drift_res, axis=-1))
        }
        
        # Devolvemos tres elementos ahora
        return drift_res, sigma_cholesky, actor_stats
        

class DirectionalActorForceAligned_new(nn.Module):
    n_elec: int
    n_nuclei: int
    hidden_dim: int = 64
    actor_amplitude: float = 0.05
    # initialize with noise

    def setup(self):
        # Tronco de características principales
        self.dense1_n = nn.Dense(features=self.hidden_dim)
        self.dense2_n = nn.Dense(features=self.hidden_dim)

        self.dense1_e = nn.Dense(features=self.hidden_dim)
        self.dense2_e = nn.Dense(features=self.hidden_dim)

        self.dense1_n_f = nn.Dense(features=self.hidden_dim)
        self.dense2_n_f = nn.Dense(features=self.hidden_dim)

        self.dense1_e_f = nn.Dense(features=self.hidden_dim)
        self.dense2_e_f = nn.Dense(features=self.hidden_dim)
        
        # Cabezas para predecir los pesos del vector de fuerza \omega(r)
        self.out_w_nuclei = nn.Dense(features=self.n_nuclei, kernel_init=nn.initializers.zeros)
        self.out_w_elec = nn.Dense(features=self.n_elec - 1, kernel_init=nn.initializers.zeros)

        # Cabezas para predecir los pesos del vector de fuerza \omega(r) (fuerzas)
        self.out_w_nuclei_f = nn.Dense(features=1, kernel_init=nn.initializers.normal(stddev=0.01))
        self.out_w_elec_f = nn.Dense(features=1, kernel_init=nn.initializers.normal(stddev=0.01))
        
        # Cabezas escalares para las varianzas principales
        # self.out_sigma_para = nn.Dense(features=1, kernel_init=nn.initializers.zeros)
        # self.out_sigma_perp = nn.Dense(features=1, kernel_init=nn.initializers.zeros)
        
        self.feat_e = nn.Dense(features=32, kernel_init=nn.initializers.lecun_normal())
        self.feat_n = nn.Dense(features=32, kernel_init=nn.initializers.lecun_normal())

        self.out_phi_amplitude = nn.Sequential([
            nn.Dense(features=self.hidden_dim, kernel_init=nn.initializers.lecun_normal()),
            # bias_init=ones permite que la multiplicación (f_n + f_e) * phi NO sea cero al inicio
            nn.Dense(features=1, kernel_init=nn.initializers.lecun_normal(), bias_init=nn.initializers.ones)
        ])

        self.sigma_perp_dense_e = nn.Dense(features=32, kernel_init=nn.initializers.zeros)
        self.sigma_perp_dense_n = nn.Dense(features=32, kernel_init=nn.initializers.zeros)

        self.out_sigma_perp = nn.Sequential([nn.Dense(features=self.hidden_dim, kernel_init=nn.initializers.zeros),
                                                        nn.Dense(features=1, kernel_init=nn.initializers.zeros)])

        self.sigma_para_dense_e = nn.Dense(features=self.hidden_dim, kernel_init=nn.initializers.zeros)
        self.sigma_para_dense_n = nn.Dense(features=self.hidden_dim, kernel_init=nn.initializers.zeros)

        self.out_sigma_para = nn.Sequential([nn.Dense(features=self.hidden_dim, kernel_init=nn.initializers.zeros),
                                                        nn.Dense(features=1, kernel_init=nn.initializers.zeros)])

    def __call__(self, walkers, elec_idx, r_nuclei, exploration_mode=True):
        # --- 1. Geometría y Vectores Unitarios ---
        r_i = walkers[:, elec_idx, :][:, None, :]
        
        vec_iI = r_i - r_nuclei[None, :, :] + 1e-8
        d_iI = jnp.linalg.norm(vec_iI, axis=-1, keepdims=True)
        u_iI = vec_iI / d_iI
        
        all_indices = jnp.arange(self.n_elec)
        shifted_indices = jnp.roll(all_indices, -elec_idx)
        others_idx = shifted_indices[1:]
        
        r_j = walkers[:, others_idx, :]
        vec_ij = r_i - r_j + 1e-8
        d_ij = jnp.linalg.norm(vec_ij, axis=-1, keepdims=True)
        u_ij = vec_ij / d_ij
        
        # Base asintótica e^{-r}
        features = jnp.concatenate([jnp.exp(-d_iI).reshape(-1, self.n_nuclei), 
                                    jnp.exp(-d_ij).reshape(-1, self.n_elec - 1)], axis=-1)

        features_e = jnp.exp(-d_ij).reshape(-1, self.n_elec - 1)
        features_n = jnp.exp(-d_iI).reshape(-1, self.n_nuclei)
        
        # --- 2. Procesamiento de la Red Neuronal ---
        x_e = nn.silu(self.dense1_e(features_e))
        x_e = nn.silu(self.dense2_e(x_e))

        x_n = nn.silu(self.dense1_n(features_n))
        x_n = nn.silu(self.dense2_n(features_n))
        
        explore_factor = jnp.where(exploration_mode, 1.0, 0.0)
        
        # --- 3. Construcción del Vector de Fuerza ---
        w_nuclei = self.out_w_nuclei(x_n)
        w_elec = self.out_w_elec(x_e)

        # extraer features para la amplitud
        features_phi_e = self.feat_e(features_e)
        features_phi_n = self.feat_n(features_n)

        features_phi = jnp.concatenate([features_phi_e, features_phi_n], axis=-1)
        
        f_nuclei = jnp.sum(w_nuclei[..., None] * u_iI, axis=1)
        f_elec = jnp.sum(w_elec[..., None] * u_ij, axis=1)
        f_total = (f_nuclei + f_elec) * self.out_phi_amplitude(features_phi)
        
        sq_norm = jnp.sum(f_total**2, axis=-1, keepdims=True)
        norm_f = jnp.sqrt(sq_norm + 1e-8)
        f_hat = f_total / norm_f
        
        drift_res = self.actor_amplitude * nn.tanh(f_total) * explore_factor

        # --- 4. Construcción Segura de la Matriz de Covarianza ---
        
        raw_para_e = self.sigma_para_dense_e(features_e)
        raw_para_n = self.sigma_para_dense_n(features_n)
        feat_para = jnp.concatenate([raw_para_e, raw_para_n], axis=-1)
        raw_para = jax.nn.softplus(self.out_sigma_para(feat_para))

        raw_perp_e = self.sigma_perp_dense_e(features_e)
        raw_perp_n = self.sigma_perp_dense_n(features_n)
        feat_perp = jnp.concatenate([raw_perp_e, raw_perp_n], axis=-1)
        raw_perp = jax.nn.softplus(self.out_sigma_perp(feat_perp))
        
        # CLIPPING ESTRICTO: Previene la explosión a infinito del RL
        sigma_para = jnp.clip(raw_para, -5.0, 1.0)
        sigma_perp = jnp.clip(raw_perp, -5.0, 1.0)

        sigma_para = jnp.exp(sigma_para)
        sigma_perp = jnp.exp(sigma_perp)
        
        # MODO EVALUACIÓN: Si está apagado, forzamos a 1.0 (Matriz Identidad para MALA clásico)
        sigma_para_sq = jnp.where(exploration_mode, sigma_para, 1.0)
        sigma_perp_sq = jnp.where(exploration_mode, sigma_perp, 1.0)

        features_e = jnp.exp(-d_ij).reshape(-1, self.n_elec - 1)
        features_n = jnp.exp(-d_iI).reshape(-1, self.n_nuclei)

        w_nuclei_f = self.forces_n_for_cov(features_n)
        w_elec_f = self.forces_e_for_cov(features_e)

        f_nuclei_cov = jnp.sum(w_nuclei_f[..., None] * u_iI, axis=1)
        f_elec_cov = jnp.sum(w_elec_f[..., None] * u_ij, axis=1)
        f_total_cov = (f_nuclei_cov + f_elec_cov) 
        
        sq_norm_cov = jnp.sum(f_total_cov**2, axis=-1, keepdims=True)
        norm_f_cov = jnp.sqrt(sq_norm_cov + 1e-8)
        f_hat_cov = f_total_cov / norm_f_cov
        
        ident = jnp.eye(3).reshape(1, 3, 3)
        f_outer = jnp.einsum('bi,bj->bij', f_hat_cov, f_hat_cov)
        
        term1 = sigma_perp_sq[..., None] * ident
        term2 = (sigma_para_sq - sigma_perp_sq)[..., None] * f_outer
        
        Sigma = term1 + term2 
        
        # Estabilizador numérico
        Sigma = Sigma + (1e-5 * ident)
        
        sigma_cholesky = jnp.linalg.cholesky(Sigma)

        # --- NUEVO: Construcción del diccionario de diagnóstico ---
        actor_stats = {
            "mean_sigma_para": jnp.mean(sigma_para_sq),
            "mean_sigma_perp": jnp.mean(sigma_perp_sq),
            "max_w_elec": jnp.max(jnp.abs(w_elec)),
            "max_w_nuc": jnp.max(jnp.abs(w_nuclei)),
            "max_w_elec_f": jnp.max(jnp.abs(w_elec_f)),
            "max_w_nuc_f": jnp.max(jnp.abs(w_nuclei_f)),
            "mean_drift_act": jnp.mean(jnp.linalg.norm(drift_res, axis=-1))
        }
        
        # Devolvemos tres elementos ahora
        return drift_res, sigma_cholesky, actor_stats

    @partial(jax.vmap, in_axes=(None, 0))
    def forces_e_for_cov(self, feature_e):
        x_e_f = nn.silu(self.dense1_e_f(feature_e))
        x_e_f = nn.silu(self.dense2_e_f(x_e_f))

        w_elec_f = self.out_w_elec_f(x_e_f)

        return w_elec_f 

    @partial(jax.vmap, in_axes=(None, 0))
    def forces_n_for_cov(self, feature_n):
        x_n_f = nn.silu(self.dense1_n_f(feature_n))
        x_n_f = nn.silu(self.dense2_n_f(x_n_f))
        
        w_nuclei_f = self.out_w_nuclei_f(x_n_f)
        return w_nuclei_f


class Actor_Huziel(nn.Module):
    """
    adapt actor "multibranch_deepset" in modular flax
    """

    n_elec: int
    n_up: int
    n_dn: int
    hidden_dim: int = 64
    actor_amplitude: float = 0.05

    def setup(self):
        # encoders
        self.local_enc = nn.Sequential([nn.Dense(32), nn.silu, nn.Dense(32), nn.silu])
        self.global_enc = nn.Sequential([nn.Dense(16), nn.silu, nn.Dense(16), nn.silu])
        
        self.same_spin_enc = nn.Sequential([nn.Dense(32), nn.silu, nn.Dense(32), nn.silu])
        self.opp_spin_enc = nn.Sequential([nn.Dense(32), nn.silu, nn.Dense(32), nn.silu])
        
        # fusion net
        self.fusion = nn.Sequential([nn.Dense(64), nn.silu, nn.Dense(64), nn.silu])
        
        # out heads
        self.out_drift = nn.Dense(1, kernel_init=nn.initializers.zeros)
        self.out_sigma = nn.Dense(1, kernel_init=nn.initializers.zeros)

    def __call__(self, walkers, elec_idx, r_nuclei, exploration_mode, grad_log_prob):
        # x y g tienen shape: [batch, N]
        x = walkers[..., 0]      
        g = grad_log_prob[..., 0] 
        
        N = self.n_elec
        
        # 1. Rama Local
        # Seleccionamos el electrón para TODO EL BATCH simultáneamente -> shape: [batch]
        x_i = x[:, elec_idx]
        g_i = g[:, elec_idx]
        
        spins = jnp.concatenate([jnp.ones(self.n_up), -jnp.ones(self.n_dn)])
        spin_i = spins[elec_idx] # Escalar
        # Expandimos el espín para que tenga el tamaño del batch -> shape: [batch]
        spin_batch = jnp.full_like(x_i, spin_i)
        
        # Centro de masa por walker -> shape: [batch]
        com = jnp.mean(x, axis=1)
        
        # Apilamos en el último eje -> shape: [batch, 4]
        local_feat = jnp.stack([
            x_i,                   
            x_i - com,     
            nn.tanh(g_i / 5.0),    
            spin_batch                 
        ], axis=-1)
        
        # 2. Rama Global
        global_feat = jnp.stack([
            N / 10.0, 
            self.n_up / max(N, 1),
            jnp.tanh(1.0), 
            jnp.tanh(1.0)  
        ]) # shape: [4]
        # Expandimos la información global a todo el batch -> shape: [batch, 4]
        global_feat = jnp.broadcast_to(global_feat, (x.shape[0], 4))
        
        # 3. Deep Sets para vecinos
        dx = x_i[:, None] - x            # shape: [batch, N]
        r = jnp.abs(dx)                  # shape: [batch, N]
        direction = jnp.sign(dx)         # shape: [batch, N]
        
        centers = jnp.linspace(0.0, 5.0, 6)
        basis = jnp.exp(-0.5 * ((r[..., None] - centers) / 0.8)**2) # shape: [batch, N, 6]
        
        pair_input = jnp.concatenate([r[..., None], direction[..., None], basis], axis=-1) # shape: [batch, N, 8]
        
        # Máscaras booleanas -> shape: [N]
        not_self = jnp.arange(N) != elec_idx 
        same_mask = (spins == spin_i) & not_self 
        opp_mask = (spins != spin_i) & not_self 
        
        # Expandimos las máscaras al batch -> shape: [batch, N]
        same_mask = jnp.broadcast_to(same_mask[None, :], (x.shape[0], N))
        opp_mask = jnp.broadcast_to(opp_mask[None, :], (x.shape[0], N))
        
        # Mensajes y Pooling (Aseguramos promediar sobre el eje N: axis=1)
        msg_same = self.same_spin_enc(pair_input) * same_mask[..., None] # shape: [batch, N, 32]
        pool_same = jnp.sum(msg_same, axis=1) / jnp.maximum(jnp.sum(same_mask, axis=1, keepdims=True), 1.0) # shape: [batch, 32]
        
        msg_opp = self.opp_spin_enc(pair_input) * opp_mask[..., None]
        pool_opp = jnp.sum(msg_opp, axis=1) / jnp.maximum(jnp.sum(opp_mask, axis=1, keepdims=True), 1.0)
        
        # 4. Fusión
        h_local = self.local_enc(local_feat)     # shape: [batch, 32]
        h_global = self.global_enc(global_feat)  # shape: [batch, 16]
        
        fused = jnp.concatenate([h_local, h_global, pool_same, pool_opp], axis=-1) # shape: [batch, 112]
        h = self.fusion(fused) # shape: [batch, 64]
        
        # 5. Salidas
        explore_factor = jnp.where(exploration_mode, 1.0, 0.0)
        
        # Salidas shape: [batch, 1]
        drift_res = self.actor_amplitude * nn.tanh(self.out_drift(h)) * explore_factor
        sigma_scale = nn.tanh(self.out_sigma(h)) * explore_factor
        
        # Métricas (aplanamos el último eje con [:, 0] para guardar escalares por walker)
        stats = {
            "d_act": jnp.abs(drift_res[:, 0]),
            "sig_scale": sigma_scale[:, 0]
        }
        
        return drift_res, sigma_scale, stats


def _default_float_dtype():
    return jnp.float64 if jax.config.x64_enabled else jnp.float32


def _init_dense(key: Array, in_dim: int, out_dim: int, zero: bool = False) -> dict:
    dtype = _default_float_dtype()
    if zero:
        w = jnp.zeros((in_dim, out_dim), dtype=dtype)
        b = jnp.zeros((out_dim,), dtype=dtype)
    else:
        limit = math.sqrt(6.0 / max(in_dim + out_dim, 1))
        w = jax.random.uniform(
            key, (in_dim, out_dim), minval=-limit, maxval=limit, dtype=dtype
        )
        b = jnp.zeros((out_dim,), dtype=dtype)
    return {"w": w, "b": b}


def init_mlp(
    key: Array,
    in_dim: int,
    hidden_dim: int,
    hidden_layers: int,
    out_dim: int,
    zero_last: bool = False,
) -> tuple[dict, ...]:
    """SiLU MLP used by actors and TD3 critics."""
    keys = jax.random.split(key, hidden_layers + 1)
    layers = []
    d = int(in_dim)
    for k in range(int(hidden_layers)):
        layers.append(_init_dense(keys[k], d, int(hidden_dim), zero=False))
        d = int(hidden_dim)
    layers.append(_init_dense(keys[-1], d, int(out_dim), zero=zero_last))
    return tuple(layers)


def mlp_apply(params: tuple[dict, ...], x: Array, final_tanh: bool = False) -> Array:
    h = jnp.asarray(x)
    for layer in params[:-1]:
        h = jax.nn.silu(h @ layer["w"] + layer["b"])
    y = h @ params[-1]["w"] + params[-1]["b"]
    return jnp.tanh(y) if final_tanh else y


def _softplus_inverse(x: Array) -> Array:
    # Stable enough for the strictly-positive widths used here.
    x = jnp.asarray(x)
    return jnp.log(jnp.expm1(jnp.maximum(x, 1e-6)))


class ActorModel:
    """Factory/dispatcher containing the available shared-electron actors.

    Available architectures
    -----------------------
    flat_mlp
        Reproduces the original actor: fixed spin-resolved pooled Gaussian-RBF
        features are concatenated with scalar features and sent through one MLP.

    multibranch_deepset
        Physics-structured actor discussed in the project:

        * local branch: selected-electron coordinate, COM-relative coordinate,
          quantum force, and selected spin;
        * global branch: N, spin fraction, interaction strength, trap frequency,
          and soft-Coulomb length;
        * separate same-spin/opposite-spin pair encoders applied to every
          neighbour and mean-pooled (Deep Sets construction);
        * fusion network;
        * separate residual-mean and proposal-scale heads.

        The radial Gaussian embedding used inside each pair encoder can be
        either fixed or trainable.  When trainable, centers and positive widths
        are part of the actor parameter tree and therefore are optimized by TD3.

    The class is intentionally stateless apart from immutable configuration;
    all trainable quantities live in the returned JAX parameter PyTree.
    """

    FLAT = "flat_mlp"
    MULTIBRANCH = "multibranch_deepset"
    AVAILABLE = (FLAT, MULTIBRANCH)

    def __init__(self, cfg, system_cfg: SystemConfig):
        self.cfg = cfg
        self.system_cfg = system_cfg
        if cfg.actor_architecture not in self.AVAILABLE:
            raise ValueError(
                f"Unknown actor architecture {cfg.actor_architecture!r}; "
                f"choose one of {self.AVAILABLE}"
            )

    @property
    def action_dim(self) -> int:
        return 1 if self.cfg.mode == "scale_only" else 2

    @property
    def max_same_neighbors(self) -> int:
        return max(max(self.system_cfg.n_up - 1, 0), max(self.system_cfg.n_down - 1, 0))

    @property
    def max_opposite_neighbors(self) -> int:
        if self.system_cfg.n_up == 0 or self.system_cfg.n_down == 0:
            return 0
        return max(self.system_cfg.n_up, self.system_cfg.n_down)

    @property
    def state_dim(self) -> int:
        if self.cfg.actor_architecture == self.FLAT:
            # Original representation: 6 scalar + four K-dimensional pooled RBF channels.
            return 6 + 4 * self.cfg.pair_feature_components

        # Structured representation stored in replay:
        # 4 local + 5 global + [r_norm, direction, mask] for each padded
        # same-spin and opposite-spin neighbour slot.
        return 4 + 5 + 3 * (self.max_same_neighbors + self.max_opposite_neighbors)

    def build_state(
        self,
        walkers: Array,
        grad_log_prob: Array,
        electron_index: int,
    ) -> Array:
        if self.cfg.actor_architecture == self.FLAT:
            return self._build_flat_state(walkers, grad_log_prob, electron_index)
        return self._build_multibranch_state(walkers, grad_log_prob, electron_index)

    def init(self, key: Array):
        if self.cfg.actor_architecture == self.FLAT:
            return init_mlp(
                key,
                self.state_dim,
                self.cfg.hidden_dim,
                self.cfg.hidden_layers,
                self.action_dim,
                zero_last=True,
            )
        return self._init_multibranch(key)

    def apply(self, params, state: Array) -> Array:
        if self.cfg.actor_architecture == self.FLAT:
            return mlp_apply(params, state, final_tanh=True)
        return self._apply_multibranch(params, state)

    # ------------------------------------------------------------------
    # Original flat actor
    # ------------------------------------------------------------------
    def _build_flat_state(self, walkers: Array, grad_log_prob: Array, electron_index: int) -> Array:
        x = jnp.asarray(walkers)
        g = jnp.asarray(grad_log_prob)
        N = self.system_cfg.n_electrons
        i = int(electron_index)
        spins = spin_labels(self.system_cfg)

        dx = x[:, i, None] - x
        r = jnp.abs(dx)
        centers = jnp.linspace(
            0.0, self.cfg.pair_feature_r_max, self.cfg.pair_feature_components
        )
        basis = jnp.exp(-0.5 * ((r[..., None] - centers) / self.cfg.pair_feature_width) ** 2)

        not_self = jnp.arange(N) != i
        same_mask = (spins == spins[i]) & not_self
        opp_mask = (spins != spins[i]) & not_self
        same_f = same_mask.astype(x.dtype)
        opp_f = opp_mask.astype(x.dtype)

        n_same = jnp.sum(same_f)
        n_opp = jnp.sum(opp_f)
        denom_same = jnp.maximum(n_same, 1.0)
        denom_opp = jnp.maximum(n_opp, 1.0)
        direction = jnp.sign(dx)

        same_even = jnp.sum(basis * same_f[None, :, None], axis=1) / denom_same
        same_odd = (
            jnp.sum(basis * direction[..., None] * same_f[None, :, None], axis=1)
            / denom_same
        )
        opp_even = jnp.sum(basis * opp_f[None, :, None], axis=1) / denom_opp
        opp_odd = (
            jnp.sum(basis * direction[..., None] * opp_f[None, :, None], axis=1)
            / denom_opp
        )

        denom_n = float(max(N - 1, 1))
        scalar = jnp.stack(
            [
                x[:, i] / self.cfg.coordinate_scale,
                (x[:, i] - jnp.mean(x, axis=1)) / self.cfg.coordinate_scale,
                jnp.tanh(g[:, i] / self.cfg.force_scale),
                jnp.full((x.shape[0],), spins[i], dtype=x.dtype),
                jnp.full((x.shape[0],), n_same / denom_n, dtype=x.dtype),
                jnp.full((x.shape[0],), n_opp / denom_n, dtype=x.dtype),
            ],
            axis=-1,
        )
        return jnp.concatenate(
            [scalar, same_even, same_odd, opp_even, opp_odd], axis=-1
        )

    # ------------------------------------------------------------------
    # Structured Deep-Sets actor
    # ------------------------------------------------------------------
    def _pair_slots(self, r: Array, dx: Array, mask: Array, max_slots: int) -> Array:
        """Return distance-sorted padded pair slots [r_norm, direction, mask].

        Sorting makes the replay representation itself invariant to relabeling
        neighbours within a spin sector.  The actor additionally applies the
        same pair encoder to every slot and pools, so its output is permutation
        invariant even without relying on slot order.
        """
        W, N = r.shape
        if max_slots == 0:
            return jnp.zeros((W, 0, 3), dtype=r.dtype)

        mask_w = jnp.broadcast_to(mask[None, :], (W, N))
        large = jnp.asarray(1e12, dtype=r.dtype)
        keys = jnp.where(mask_w, r, large)
        order = jnp.argsort(keys, axis=1)[:, :max_slots]
        rr = jnp.take_along_axis(r, order, axis=1)
        dd = jnp.take_along_axis(dx, order, axis=1)
        mm = jnp.take_along_axis(mask_w, order, axis=1)

        rr = jnp.where(mm, rr / self.cfg.pair_feature_r_max, 0.0)
        direction = jnp.where(mm, jnp.sign(dd), 0.0)
        return jnp.stack([rr, direction, mm.astype(r.dtype)], axis=-1)

    def _build_multibranch_state(
        self, walkers: Array, grad_log_prob: Array, electron_index: int
    ) -> Array:
        x = jnp.asarray(walkers)
        g = jnp.asarray(grad_log_prob)
        i = int(electron_index)
        N = self.system_cfg.n_electrons
        spins = spin_labels(self.system_cfg)

        # Dynamic local information.  The quantum force is already computed by
        # MALA, so exposing it to the actor has negligible extra physics cost.
        local = jnp.stack(
            [
                x[:, i] / self.cfg.coordinate_scale,
                (x[:, i] - jnp.mean(x, axis=1)) / self.cfg.coordinate_scale,
                jnp.tanh(g[:, i] / self.cfg.force_scale),
                jnp.full((x.shape[0],), spins[i], dtype=x.dtype),
            ],
            axis=-1,
        )

        # Static system context.  These values do not help distinguish steps
        # within one run, but allow one shared policy to condition on different
        # systems in transfer/generalization experiments.
        global_vec = jnp.asarray(
            [
                N / 10.0,
                self.system_cfg.n_up / max(N, 1),
                math.tanh(float(self.system_cfg.interaction_strength)),
                math.tanh(float(self.system_cfg.omega)),
                math.tanh(float(self.system_cfg.softening)),
            ],
            dtype=x.dtype,
        )
        global_context = jnp.broadcast_to(global_vec[None, :], (x.shape[0], 5))

        dx = x[:, i, None] - x
        r = jnp.abs(dx)
        not_self = jnp.arange(N) != i
        same_mask = (spins == spins[i]) & not_self
        opp_mask = (spins != spins[i]) & not_self

        same_slots = self._pair_slots(r, dx, same_mask, self.max_same_neighbors)
        opp_slots = self._pair_slots(r, dx, opp_mask, self.max_opposite_neighbors)

        return jnp.concatenate(
            [
                local,
                global_context,
                same_slots.reshape((x.shape[0], -1)),
                opp_slots.reshape((x.shape[0], -1)),
            ],
            axis=-1,
        )

    def _init_radial_embedding(self, key: Array) -> dict:
        K = int(self.cfg.pair_feature_components)
        width_norm = max(
            float(self.cfg.pair_feature_width) / float(self.cfg.pair_feature_r_max),
            1e-3,
        )
        if self.cfg.trainable_embedding:
            # Interior initialization allows sigmoid-parameterized centers to
            # move in both directions while remaining in [0, 1].
            centers = jnp.linspace(0.05, 0.95, K, dtype=_default_float_dtype())
            center_raw = jnp.log(centers) - jnp.log1p(-centers)
            width_raw = jnp.full(
                (K,), _softplus_inverse(jnp.asarray(width_norm, dtype=centers.dtype)),
                dtype=centers.dtype,
            )
            return {"center_raw": center_raw, "width_raw": width_raw}
        return {}

    def _embedding_values(self, r_norm: Array, params: dict) -> Array:
        K = int(self.cfg.pair_feature_components)
        if self.cfg.trainable_embedding:
            centers = jax.nn.sigmoid(params["center_raw"])
            widths = jax.nn.softplus(params["width_raw"]) + 1e-4
        else:
            centers = jnp.linspace(0.0, 1.0, K, dtype=r_norm.dtype)
            widths = jnp.full(
                (K,),
                max(
                    float(self.cfg.pair_feature_width)
                    / float(self.cfg.pair_feature_r_max),
                    1e-4,
                ),
                dtype=r_norm.dtype,
            )
        return jnp.exp(-0.5 * ((r_norm[..., None] - centers) / widths) ** 2)

    def _init_multibranch(self, key: Array):
        keys = jax.random.split(key, 8)
        pair_in = 2 + int(self.cfg.pair_feature_components)  # r_norm, direction, RBFs

        params = {
            "local_encoder": init_mlp(
                keys[0], 4, self.cfg.local_embedding_dim,
                self.cfg.branch_hidden_layers, self.cfg.local_embedding_dim,
            ),
            "global_encoder": init_mlp(
                keys[1], 5, self.cfg.global_embedding_dim,
                self.cfg.branch_hidden_layers, self.cfg.global_embedding_dim,
            ),
            "same_pair_encoder": init_mlp(
                keys[2], pair_in, self.cfg.pair_embedding_dim,
                self.cfg.branch_hidden_layers, self.cfg.pair_embedding_dim,
            ),
            "opposite_pair_encoder": init_mlp(
                keys[3], pair_in, self.cfg.pair_embedding_dim,
                self.cfg.branch_hidden_layers, self.cfg.pair_embedding_dim,
            ),
            "radial_embedding": self._init_radial_embedding(keys[4]),
        }
        fusion_in = (
            self.cfg.local_embedding_dim
            + self.cfg.global_embedding_dim
            + 2 * self.cfg.pair_embedding_dim
        )
        params["fusion"] = init_mlp(
            keys[5], fusion_in, self.cfg.fusion_dim,
            self.cfg.fusion_hidden_layers, self.cfg.fusion_dim,
        )
        if self.cfg.mode != "scale_only":
            params["mean_head"] = init_mlp(
                keys[6], self.cfg.fusion_dim, self.cfg.head_hidden_dim,
                self.cfg.head_hidden_layers, 1, zero_last=True,
            )
        params["scale_head"] = init_mlp(
            keys[7], self.cfg.fusion_dim, self.cfg.head_hidden_dim,
            self.cfg.head_hidden_layers, 1, zero_last=True,
        )
        return params

    def _decode_multibranch_state(self, state: Array):
        s = jnp.asarray(state)
        p = 0
        local = s[..., p : p + 4]
        p += 4
        global_context = s[..., p : p + 5]
        p += 5

        ns = self.max_same_neighbors
        no = self.max_opposite_neighbors
        same = s[..., p : p + 3 * ns].reshape(s.shape[:-1] + (ns, 3))
        p += 3 * ns
        opp = s[..., p : p + 3 * no].reshape(s.shape[:-1] + (no, 3))
        return local, global_context, same, opp

    def _encode_pair_branch(self, slots: Array, encoder_params, radial_params) -> Array:
        # slots: (..., n_slots, 3) = [r_norm, direction, mask]
        if slots.shape[-2] == 0:
            return jnp.zeros(slots.shape[:-2] + (self.cfg.pair_embedding_dim,), dtype=slots.dtype)
        r_norm = slots[..., 0]
        direction = slots[..., 1]
        mask = slots[..., 2]
        radial = self._embedding_values(r_norm, radial_params)
        pair_input = jnp.concatenate(
            [r_norm[..., None], direction[..., None], radial], axis=-1
        )
        messages = mlp_apply(encoder_params, pair_input, final_tanh=False)
        messages = messages * mask[..., None]
        denom = jnp.maximum(jnp.sum(mask, axis=-1, keepdims=True), 1.0)
        return jnp.sum(messages, axis=-2) / denom

    def _apply_multibranch(self, params, state: Array) -> Array:
        local, global_context, same, opp = self._decode_multibranch_state(state)
        h_local = mlp_apply(params["local_encoder"], local, final_tanh=False)
        h_global = mlp_apply(params["global_encoder"], global_context, final_tanh=False)
        h_same = self._encode_pair_branch(
            same, params["same_pair_encoder"], params["radial_embedding"]
        )
        h_opp = self._encode_pair_branch(
            opp, params["opposite_pair_encoder"], params["radial_embedding"]
        )
        fused = jnp.concatenate([h_local, h_same, h_opp, h_global], axis=-1)
        h = mlp_apply(params["fusion"], fused, final_tanh=False)

        scale = mlp_apply(params["scale_head"], h, final_tanh=True)[..., 0]
        if self.cfg.mode == "scale_only":
            return scale[..., None]
        mean = mlp_apply(params["mean_head"], h, final_tanh=True)[..., 0]
        return jnp.stack([mean, scale], axis=-1)

    def embedding_parameters_physical(self, params) -> dict:
        """Return human-readable radial centers/widths for diagnostics."""
        if self.cfg.actor_architecture != self.MULTIBRANCH:
            return {}
        K = int(self.cfg.pair_feature_components)
        if self.cfg.trainable_embedding:
            radial = params["radial_embedding"]
            centers_norm = jax.nn.sigmoid(radial["center_raw"])
            widths_norm = jax.nn.softplus(radial["width_raw"]) + 1e-4
        else:
            centers_norm = jnp.linspace(0.0, 1.0, K, dtype=_default_float_dtype())
            widths_norm = jnp.full(
                (K,),
                float(self.cfg.pair_feature_width) / float(self.cfg.pair_feature_r_max),
                dtype=_default_float_dtype(),
            )
        return {
            "centers": centers_norm * self.cfg.pair_feature_r_max,
            "widths": widths_norm * self.cfg.pair_feature_r_max,
        }
