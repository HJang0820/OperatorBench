import numpy as np

from operatorbench.data_generation.darcy_fd import solve_darcy_fd
from operatorbench.data_generation.forcing import (
    constant_source,
    sine_source,
)

def main():
    # 1. Grid and coefficient field
    N = 33
    a = np.ones((N, N), dtype=np.float64)

    # 2. A scalar and an equivalent constant array must agree.
    u_scalar = solve_darcy_fd(a, f_value=1.0)

    f_constant = constant_source(N, value=1.0)
    u_constant_array = solve_darcy_fd(a, f_value=f_constant)

    np.testing.assert_allclose(
        u_scalar,
        u_constant_array,
        rtol=1e-12,
        atol=1e-14,
    )

    print("Scalar/constant-array agreement passed.")

    # 3. Solve the same problem on successively finer grids.
    grid_sizes = (17, 33, 65)
    errors = []
    grid_spacings = []

    for N in grid_sizes:
        h = 1.0 / (N - 1)

        # Rebuild coordinates and fields for the current grid.
        x = np.linspace(0.0, 1.0, N)
        y = np.linspace(0.0, 1.0, N)
        X, Y = np.meshgrid(x, y, indexing="ij")

        a = np.ones((N, N), dtype=np.float64)

        u_exact = np.sin(np.pi * X) * np.sin(2.0 * np.pi * Y)

        f_grid = sine_source(
            N,
            amplitude=5.0 * np.pi**2,
        )

        # Check that the generator reproduces the original source.
        np.testing.assert_allclose(
            f_grid,
            5.0 * np.pi**2 * u_exact,
            rtol=1e-12,
            atol=1e-12,
        )

        u_numeric = solve_darcy_fd(a, f_value=f_grid)

        # Check output shape, finite values, and boundary conditions.
        assert u_numeric.shape == (N, N)
        assert np.isfinite(u_numeric).all()
        assert np.all(u_numeric[[0, -1], :] == 0.0)
        assert np.all(u_numeric[:, [0, -1]] == 0.0)

        relative_error = (
            np.linalg.norm(u_numeric - u_exact)
            / np.linalg.norm(u_exact)
        )

        assert np.isfinite(relative_error)
        assert relative_error > 0.0

        errors.append(relative_error)
        grid_spacings.append(h)

        print(
            f"N={N:2d}, h={h:.6f}, "
            f"relative L2 error={relative_error:.6e}"
        )

    # 4. Measure the convergence order between adjacent grids.
    for k in range(1, len(grid_sizes)):
        coarse_error = errors[k - 1]
        fine_error = errors[k]

        coarse_h = grid_spacings[k - 1]
        fine_h = grid_spacings[k]

        order = (
            np.log(coarse_error / fine_error)
            / np.log(coarse_h / fine_h)
        )

        print(
            f"{grid_sizes[k - 1]} -> {grid_sizes[k]}: "
            f"observed order={order:.4f}"
        )

        assert fine_error < coarse_error, (
            "Error did not decrease under grid refinement."
        )

        # Expected for this smooth, constant-coefficient test.
        assert 1.8 < order < 2.2, (
            f"Unexpected convergence order: {order:.4f}"
        )

    print("Grid refinement checks passed.")

if __name__ == "__main__":
    main()