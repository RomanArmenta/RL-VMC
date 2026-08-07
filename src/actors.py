from flax import linen as nn
import jax.numpy as jnp
import jax

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
            "max_w_nuclei": jnp.max(jnp.abs(w_nuclei)),
            "mean_drift_act": jnp.mean(jnp.linalg.norm(drift_res, axis=-1))
        }
        
        # Devolvemos tres elementos ahora
        return drift_res, sigma_cholesky, actor_stats
        