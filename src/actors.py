from flax import linen as nn
import jax.numpy as jnp

class DirectionalActor(nn.Module):
    n_elec: int
    n_nuclei: int
    hidden_dim: int = 64

    def setup(self):
        self.dense1 = nn.Dense(features=self.hidden_dim)
        self.dense2 = nn.Dense(features=self.hidden_dim)
        self.out_w_nuclei = nn.Dense(features=self.n_nuclei, kernel_init=nn.initializers.zeros)
        self.out_w_elec = nn.Dense(features=self.n_elec - 1, kernel_init=nn.initializers.zeros)
        self.out_sigma = nn.Dense(features=3, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros)

    def __call__(self, walkers, elec_idx, r_nuclei, exploration_mode=True):
        r_i = walkers[:, elec_idx, :][:, None, :]
        
        vec_iI = r_i - r_nuclei[None, :, :]
        d_iI = jnp.linalg.norm(vec_iI, axis=-1, keepdims=True)
        u_iI = vec_iI / (d_iI + 1e-8)
        
        all_indices = jnp.arange(self.n_elec)
        shifted_indices = jnp.roll(all_indices, -elec_idx)
        others_idx = shifted_indices[1:]
        
        r_j = walkers[:, others_idx, :]
        vec_ij = r_i - r_j
        d_ij = jnp.linalg.norm(vec_ij, axis=-1, keepdims=True)
        u_ij = vec_ij / (d_ij + 1e-8)
        
        features = jnp.concatenate([jnp.exp(-d_iI).reshape(-1, self.n_nuclei), 
                                    jnp.exp(-d_ij).reshape(-1, self.n_elec - 1)], axis=-1)
        
        x = nn.relu(self.dense1(features))
        x = nn.relu(self.dense2(x))
        
        w_nuclei = self.out_w_nuclei(x)
        w_elec = self.out_w_elec(x)
        
        drift_nuclei = jnp.sum(w_nuclei[..., None] * u_iI, axis=1)
        drift_elec = jnp.sum(w_elec[..., None] * u_ij, axis=1)
        
        explore_factor = jnp.where(exploration_mode, 1.0, 0.0)
        drift_res = 0.5 * nn.tanh(drift_nuclei + drift_elec) * explore_factor

        sigma_raw = self.out_sigma(x)
        sigma_scale = nn.tanh(sigma_raw)
        
        return drift_res, sigma_scale