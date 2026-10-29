# Guided examples

Run these commands from the package directory after `python -m pip install -e .`.

## 1. Establish the noninteracting reference

```bash
uhf1d --method uhf --n-up 2 --n-down 1 --g 0 --n-basis 6 \
  --quadrature-order 40 --output-dir results/noninteracting
```

The exact energy is

```text
E = omega/2 * (N_up^2 + N_down^2) = 2.5 omega.
```

At `g=0`, the coefficient matrices are the occupied HO orbitals. This is the
best first check of spin counts, units, basis ordering, and installation.

## 2. Compare UHF and CASSCF for two interacting electrons

```bash
uhf1d --method uhf --n-up 1 --n-down 1 --g 1 --softening 0.5 \
  --n-basis 6 --quadrature-order 48 --output-dir results/n2_uhf

uhf1d --method casscf --config examples/casscf_n2.json \
  --output-dir results/n2_casscf
```

`CAS(2,4)` here means two active electrons distributed in four active spatial
orbitals. With fixed `N_up=N_down=1`, its determinant dimension is
`choose(4,1)*choose(4,1)=16`. For the shipped settings the CASSCF energy is
about `2.12640915`, below the UHF value `2.19036436`.

Inspect `ci_weights.png`: a dominant first bar indicates a mostly
single-reference state; several appreciable bars indicate stronger
multideterminant character.

## 3. Export only the four leading determinants

```bash
uhf1d --method casscf --config examples/casscf_n2_compact.json \
  --output-dir results/n2_top4
```

This still performs the full 16-determinant CAS calculation, but exports a
normalized four-determinant wavefunction in `selected_ci_*`. For the shipped
example those four determinants retain approximately `0.99810313` of the
full-CI norm. Compare several sizes systematically:

```bash
for number in 1 2 4 8 16; do
  uhf1d --method casscf --config examples/casscf_n2.json \
    --n-final-determinants "$number" --output-dir "results/n2_top${number}"
done
```

The full CASSCF calculation is repeated in this simple loop. Since determinant
selection is post-processing, one production run can instead be loaded in
Python and sliced at several thresholds.

## 4. Four electrons with a frozen core

```bash
uhf1d --method casscf --config examples/casscf_n4_frozen_core.json \
  --output-dir results/n4_frozen_core
```

Here `N_up=N_down=2` and `n_core=1`. The lowest optimized spatial orbital is
doubly occupied, leaving one up and one down electron in three active orbitals:
`CAS(2,3)` with nine determinants. The final export keeps the six largest.
Gray curves in the determinant mosaic represent the common doubly occupied
core orbital.

## 5. Active-space convergence

Keep the one-particle basis fixed while increasing only the active space:

```bash
uhf1d --method casscf --config examples/casscf_n2.json \
  --n-active-orbitals 2 --output-dir results/cas22
uhf1d --method casscf --config examples/casscf_n2.json \
  --n-active-orbitals 3 --output-dir results/cas23
uhf1d --method casscf --config examples/casscf_n2.json \
  --n-active-orbitals 4 --output-dir results/cas24
```

The full-CAS energy should not increase as the active space grows, apart from
small numerical optimizer tolerances. After choosing an active space, increase
`n_basis` and `quadrature_order` separately to check basis and integral
convergence.

## 6. Read the compact state

```python
import json
import numpy as np

with np.load("results/n2_top4/casscf_state.npz") as state:
    orbitals = state["orbital_coefficients"]
    active_orbitals = state["active_orbital_coefficients"]
    coefficients = state["selected_ci_coefficients"]
    determinants = state["selected_ci_determinants"].astype(int)
    retained_norm = float(state["selected_ci_retained_norm"])
    metadata = json.loads(str(state["metadata_json"]))

print(coefficients)
print(determinants)
print(retained_norm)
```

For `M` active orbitals, bits `0..M-1` specify occupied up-spin active
orbitals; bits `M..2M-1` specify occupied down-spin active orbitals. The spatial
orbital values needed for a Slater determinant are
`ho_basis(x, n_basis, omega) @ orbital_coefficients`.

## Reading the determinant mosaic

Each tile is one selected determinant:

- tiles are ordered by decreasing `|c_I|`;
- the title gives the renormalized selected coefficient and weight `c_I^2`;
- `a0`, `a1`, ... label active orbitals;
- solid lines are occupied by up-spin electrons;
- dashed lines are occupied by down-spin electrons;
- the lower-left annotation lists the occupied active-orbital indices.

The absolute sign of one CI coefficient depends on orbital phase conventions;
relative signs between determinants are physically meaningful.
