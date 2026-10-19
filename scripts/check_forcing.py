import numpy as np

from operatorbench.data_generation.forcing import (
    constant_source,
    sine_source,
)


def main():
    # 1. Check different grid sizes and constant values.
    for N in (3, 5, 33):
        for value in (0.0, 1.0, -2.0, 2.5):
            f = constant_source(N, value=value)

            assert f.shape == (N, N)
            assert f.dtype == np.float64
            assert np.isfinite(f).all()
            assert np.all(f == value)

    # Check the default source value.
    f_default = constant_source(5)
    assert np.all(f_default == 1.0)

    print("Shape, dtype, values, and default checks passed.")

    # 2. These inputs should raise ValueError.
    invalid_inputs = (
        (2, 1.0),
        (5.0, 1.0),
        (True, 1.0),
        (5, np.nan),
        (5, np.inf),
        (5, [1.0, 2.0]),
    )

    for N, value in invalid_inputs:
        try:
            constant_source(N, value=value)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Invalid input was accepted: N={N}, value={value}"
            )

    print("Invalid-input checks passed.")

    # 3. Display one small example.
    print("Example: f(x, y) = 2.5 on a 3 x 3 grid")
    print(constant_source(3, value=2.5))

    print("Constant source checks passed.")
    # 4. Check the sine source generator.
    N = 17

    for amplitude in (0.0, 1.0, -2.0, 5.0 * np.pi**2):
        f = sine_source(N, amplitude=amplitude)

        assert f.shape == (N, N)
        assert f.dtype == np.float64
        assert np.isfinite(f).all()

        # x[8] = 0.5, y[4] = 0.25: f = amplitude.
        np.testing.assert_allclose(
            f[8, 4],
            amplitude,
            rtol=1e-12,
            atol=1e-12,
        )

        # x[8] = 0.5, y[12] = 0.75: f = -amplitude.
        np.testing.assert_allclose(
            f[8, 12],
            -amplitude,
            rtol=1e-12,
            atol=1e-12,
        )

        # This particular sine pattern vanishes on the boundary.
        np.testing.assert_allclose(
            f[[0, -1], :],
            0.0,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            f[:, [0, -1]],
            0.0,
            rtol=0.0,
            atol=1e-12,
        )

    print("Sine source checks passed.")

if __name__ == "__main__":
    main()