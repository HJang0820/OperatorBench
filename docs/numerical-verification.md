# Numerical verification

### Source-array support and grid refinement

Run the verification script from the repository root:

```bash
python -m scripts.check_darcy_source
```

The script first checks that a scalar source `f = 1` and an equivalent
constant source array produce matching numerical solutions.

It then solves a manufactured-solution problem on the unit square with
homogeneous Dirichlet boundary conditions:

$$
a(x,y)=1,
$$

$$
u_{\mathrm{exact}}(x,y)=\sin(\pi x)\sin(2\pi y),
$$

$$
f(x,y)=5\pi^2\sin(\pi x)\sin(2\pi y).
$$

The same continuous problem is evaluated on each grid.
Each grid contains `N x N` nodes, including boundary nodes, with
spacing `h = 1 / (N - 1)`.

The relative L2 error is computed from the sampled solution arrays:

$$
E_h =
\frac{\|u_h-u_{\mathrm{exact}}\|_2}
{\|u_{\mathrm{exact}}\|_2}.
$$

### Results

| Grid size | Grid spacing | Relative L2 error | Observed order |
| --- | --- | --- | --- |
| 17 x 17 | 1/16 | 1.098931e-02 | — |
| 33 x 33 | 1/32 | 2.734955e-03 | 2.0065 |
| 65 x 65 | 1/64 | 6.829684e-04 | 2.0016 |

The observed order compares each grid with the preceding coarser grid:

$$
p =
\frac{\log(E_{\mathrm{coarse}}/E_{\mathrm{fine}})}
{\log(h_{\mathrm{coarse}}/h_{\mathrm{fine}})}.
$$

The error decreases by approximately a factor of four when the grid
spacing is halved. This is consistent with second-order convergence
for this smooth, constant-coefficient problem.

Successful execution ends with:

```text
Grid refinement checks passed.
```

### Scope and limitations

These results concern finite-difference discretization error, not
DeepONet or FNO prediction error.

The test verifies source-array handling and grid convergence for the
specified manufactured solution. It does not establish reference-solution
accuracy for discontinuous coefficient fields or validate 421 x 421
reference datasets. Those checks remain separate tasks.