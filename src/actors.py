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
        # reduce to 1D
        x = walkers[..., 0]      # [N]
        g = grad_log_prob[..., 0] # [N]
        
        N = self.n_elec
        
        # local branch (single electron characteristics)
        x_i = x[elec_idx]
        g_i = g[elec_idx]
        
        # generate spin vector (+1 for ip -1 for down)
        spins = jnp.concatenate([jnp.ones(self.n_up), -jnp.ones(self.n_dn)])
        spin_i = spins[elec_idx]
        
        local_feat = jnp.stack([
            x_i,                   # position
            x_i - jnp.mean(x),     # distance to mass center
            nn.tanh(g_i / 5.0),    # quantum force with clipping
            spin_i                 # spin
        ])
        
        # global branch - whole system characteristic vbector
        global_feat = jnp.stack([
            N / 10.0, 
            self.n_up / max(N, 1),
            jnp.tanh(1.0), # interactions (hardcoded)
            jnp.tanh(1.0)  # trap freq
        ])
        
        # Deep Sets for neighbors
        dx = x_i - x
        r = jnp.abs(dx)
        direction = jnp.sign(dx)
        
        # Gaussians for basis
        centers = jnp.linspace(0.0, 5.0, 6)
        basis = jnp.exp(-0.5 * ((r[..., None] - centers) / 0.8)**2)
        
        # inputs for pairs: [dist, direction, base_rbf]
        pair_input = jnp.concatenate([r[..., None], direction[..., None], basis], axis=-1)
        
        # masks
        not_self = jnp.arange(N) != elec_idx
        same_mask = (spins == spin_i) & not_self
        opp_mask = (spins != spin_i) & not_self
        
        # apply encoders and take mean
        msg_same = self.same_spin_enc(pair_input) * same_mask[..., None]
        pool_same = jnp.sum(msg_same, axis=0) / jnp.maximum(jnp.sum(same_mask), 1.0)
        
        msg_opp = self.opp_spin_enc(pair_input) * opp_mask[..., None]
        pool_opp = jnp.sum(msg_opp, axis=0) / jnp.maximum(jnp.sum(opp_mask), 1.0)
        
        # fusion
        h_local = self.local_enc(local_feat)
        h_global = self.global_enc(global_feat)
        
        fused = jnp.concatenate([h_local, h_global, pool_same, pool_opp], axis=-1)
        h = self.fusion(fused)
        
        # out
        explore_factor = jnp.where(exploration_mode, 1.0, 0.0)
        
        # actor return drift and scale, we make it 1D
        drift_res = self.actor_amplitude * nn.tanh(self.out_drift(h)) * explore_factor
        sigma_scale = nn.tanh(self.out_sigma(h)) * explore_factor
        
        stats = {
            "d_act": jnp.abs(drift_res[0]),
            "sig_scale": sigma_scale[0]
        }
        
        # (batch, N_elec, 1) to match 3D sampler
        return drift_res, sigma_scale, stats