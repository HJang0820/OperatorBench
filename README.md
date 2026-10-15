# OperatorBench

OperatorBench is a research project for reproducible comparisons of
DeepONet and Fourier Neural Operator (FNO) on PDE operator-learning tasks.

The immediate focus is a complete 2D Darcy workflow:
data generation, storage, preprocessing, model training, and evaluation.
Burgers experiments are planned as a subsequent extension.

## Current status

Preliminary Darcy experiments were conducted in Jupyter notebooks.
The existing implementation is being migrated into reusable Python modules.

The coefficient generator, finite-difference solver, and diagnostic
plotting script have been extracted and passed basic smoke checks.
The complete modular training and evaluation pipeline is not yet available.

## Darcy problem

We consider the steady elliptic problem

$$
-\nabla \cdot (a(x)\nabla u(x)) = f(x),
\qquad x \in (0,1)^2,
$$

with homogeneous Dirichlet boundary conditions:

$$
u(x) = 0, \qquad x \in \partial\Omega.
$$

Here, `a` is the positive coefficient field, `f` is the source term,
and `u` is the solution field.

### Current diagnostic configuration

- Model input for the initial task: coefficient field `a`.
- Source term: `f = 1`.
- Coefficient values: `3` and `12`.
- Coefficient generation: thresholding a random cosine-basis expansion.
- The constant mode of the underlying random field is set to zero.
- Grid: `85 x 85` nodes, including boundary nodes.
- Boundary conditions: zero Dirichlet conditions on all four sides.
- Solver: flux-form five-point finite differences.
- Face coefficients: arithmetic averages of adjacent nodal coefficients.
- Linear system: solved using SciPy's sparse direct solver.

This is a locally generated baseline dataset configuration.
Exact equivalence to the original public FNO dataset has not been established.

### Planned input configurations

| Task | Model input | Fixed information |
| --- | --- | --- |
| `a_to_u` | Coefficient field `a` | A prescribed source, initially `f = 1` |
| `f_to_u` | Source field `f` | One prescribed coefficient field shared across samples |
| `af_to_u` | Coefficient and source fields `(a, f)` | Domain and boundary conditions |

The current diagnostic demonstrates the constant-source configuration.
Spatially varying source fields and the additional learning tasks are planned.

### Planned coefficient distributions

- Binary, discontinuous coefficients: current implementation.
- Positive lognormal coefficients with configurable spatial regularity: planned.

For lognormal coefficients, spatial regularity will be controlled by
the underlying Gaussian field. Exponentiation alone does not specify smoothness.

## Run the current Darcy diagnostic

Run the following commands from the repository root in your Python environment.

Install the diagnostic dependencies:

```bash
python -m pip install numpy scipy matplotlib
```

Run the checks and generate the figure:

```bash
python -m scripts.check_darcy
```

The script generates one coefficient field, computes its finite-difference
solution, checks basic properties, evaluates the discrete residual,
and saves a diagnostic figure.

Successful execution includes:

```text
Basic solver checks passed.
```

The figure is saved to:

```text
docs/figures/darcy_sample_N85_seed2022.png
```

### Example coefficient and finite-difference solution

![Binary Darcy coefficient and finite-difference solution](docs/figures/darcy_sample_N85_seed2022.png)

This figure shows a numerical solution, not a neural-network prediction.
A small discrete residual checks consistency with the discrete equations;
it does not establish continuum accuracy or grid convergence.

## Modular design

The tree below includes both existing components and planned modules.
Files marked "planned" are not yet available.

```text
OperatorBench/
├── operatorbench/
│   ├── __init__.py
│   ├── data_generation/
│   │   ├── __init__.py
│   │   ├── fields.py             # Coefficient generation
│   │   ├── darcy_fd.py           # Darcy finite-difference solver
│   │   ├── forcing.py            # Planned: source generation
│   │   └── darcy_tasks.py        # Planned: task configuration
│   ├── data/                    # Planned
│   │   ├── datasets.py
│   │   └── preprocessing.py
│   ├── models/                  # Planned notebook migration
│   │   ├── deeponet.py
│   │   └── fno.py
│   ├── training.py              # Planned
│   └── evaluation.py            # Planned
├── scripts/
│   ├── __init__.py
│   └── check_darcy.py
├── docs/
│   └── figures/
│       └── darcy_sample_N85_seed2022.png
├── configs/                     # Planned
├── tests/                       # Planned
└── notebooks/                   # Planned
```

### Design principles

1. Separate numerical data generation from model training.
2. Preserve `a`, `f`, `u`, coordinates, boundary conditions, and generation metadata.
   Fixed fields may be stored as shared metadata rather than duplicated arrays.
3. Select the model inputs through task configuration.
4. Give DeepONet and FNO the same physical input information and data splits.
5. Fit normalization statistics using training data only.
6. Select checkpoints using validation data and reserve test data for final evaluation.
7. Separate code migration from changes to numerical methods and model architectures.
8. Record dependency versions, random seeds, configurations, and evaluation rules.

## Darcy roadmap

- [x] Extract the binary coefficient generator.
- [x] Extract the constant-source Darcy solver.
- [x] Add basic checks and a diagnostic figure.
- [ ] Verify numerical equivalence between notebook and modular implementations.
- [ ] Support spatially varying source fields in the solver.
- [ ] Add configurable coefficient and source generators.
- [ ] Define and implement the common dataset format.
- [ ] Add data loading and training-only normalization.
- [ ] Migrate DeepONet and FNO.
- [ ] Implement shared training, validation, and evaluation procedures.
- [ ] Report relative L2 error and maximum absolute error.
- [ ] Validate numerical accuracy using manufactured solutions and grid refinement.
- [ ] Validate higher-resolution reference generation, including 421 x 421 grids.
- [ ] Document reproducible Darcy baseline results.

Burgers support will follow the Darcy milestone.

## References

- [Fourier Neural Operator](https://arxiv.org/abs/2010.08895)
- [PDEBench](https://github.com/pdebench/PDEBench)