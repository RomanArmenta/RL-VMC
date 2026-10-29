import json
import numpy as np
from uhf1d import CASSCFConfig, UHFConfig, run_casscf, run_uhf


def test_noninteracting_limit(tmp_path):
    cfg = UHFConfig(n_up=3, n_down=2, g=0.0, n_basis=8, quadrature_order=40)
    r = run_uhf(cfg)
    exact = cfg.omega * (cfg.n_up**2 + cfg.n_down**2) / 2
    assert r.converged
    assert abs(r.energies["total"] - exact) < 1e-11
    assert abs(r.energies["hartree_coulomb"]) < 1e-14
    assert np.allclose(np.abs(r.coefficients_up[:cfg.n_up, :cfg.n_up]), np.eye(cfg.n_up), atol=1e-10)
    assert np.allclose(r.coefficients_up.T @ r.overlap @ r.coefficients_up, np.eye(cfg.n_basis), atol=1e-10)
    path = tmp_path / "state.npz"; r.save(path)
    with np.load(path) as z:
        meta = json.loads(str(z["metadata_json"])); assert meta["format_version"] == 1
        assert z["occupied_coefficients_up"].shape == (cfg.n_basis, cfg.n_up)


def test_finite_g_smoke():
    cfg = UHFConfig(n_up=1, n_down=1, g=0.5, softening=0.5, n_basis=6,
                    quadrature_order=36, max_iter=150, mixing=0.5)
    r = run_uhf(cfg)
    assert r.converged and np.isfinite(r.energies["total"])
    assert r.energies["hartree_coulomb"] > 0 and r.energies["exchange"] < 0


def test_casscf_noninteracting_limit(tmp_path):
    cfg = CASSCFConfig(n_up=1, n_down=1, g=0.0, n_basis=3, quadrature_order=24,
                       n_core=0, n_active_orbitals=2, casscf_max_iter=5)
    r = run_casscf(cfg)
    assert r.converged and abs(r.energy - 1.0) < 1e-11
    assert np.max(np.abs(r.orbital_coefficients.T @ r.overlap @ r.orbital_coefficients - np.eye(3))) < 1e-10
    path = tmp_path / "cas.npz"; r.save(path)
    with np.load(path) as z:
        assert z["active_orbital_coefficients"].shape == (3, 2)
        assert len(z["ci_coefficients"]) == 4


def test_casscf_finite_g_smoke():
    cfg = CASSCFConfig(n_up=1, n_down=1, g=0.4, softening=0.7,
                       n_basis=2, quadrature_order=24, n_core=0,
                       n_active_orbitals=2, casscf_max_iter=5)
    r = run_casscf(cfg)
    assert r.converged and np.isfinite(r.energy)


def test_selected_determinant_export(tmp_path):
    cfg = CASSCFConfig(n_up=1, n_down=1, g=0.3, softening=0.8,
                       n_basis=2, quadrature_order=24, n_active_orbitals=2,
                       n_final_determinants=2, casscf_max_iter=3)
    r = run_casscf(cfg); c, d, indices, norm = r.selected_ci()
    assert len(c) == len(d) == len(indices) == 2
    assert np.all(np.abs(r.ci_coefficients[indices[:-1]]) >= np.abs(r.ci_coefficients[indices[1:]]))
    assert np.isclose(np.dot(c, c), 1.0) and 0 < norm <= 1
    path = tmp_path/"selected.npz"; r.save(path)
    with np.load(path) as z:
        assert z["selected_ci_coefficients"].shape == (2,)
