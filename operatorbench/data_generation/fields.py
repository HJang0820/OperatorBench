import numpy as np

def sample_fno_darcy_coefficients(n_samples, N, n_modes=None, seed=None):
    rng = np.random.default_rng(seed)
    if n_modes is None:
        n_modes = N

    x = np.linspace(0, 1, N)
    k = np.arange(n_modes)

    c = np.ones(n_modes)
    c[1:] = np.sqrt(2)
    Phi = np.cos(np.pi * x[:, None] * k[None, :])
    Phi = Phi * c[None, :]

    K1, K2 = np.meshgrid(k, k, indexing="ij")
    lam = np.pi**2 * (K1**2 + K2**2)
    std = 1 / (lam + 9)
    # 기존 구현의 상수 모드 제거를 유지한다.
    fields = []
    for _ in range(n_samples):
        xi = rng.standard_normal((n_modes, n_modes))
        coeff = xi * std
        
        coeff[0, 0] = 0.0
        g = (Phi @ coeff) @ Phi.T
        fields.append(np.where(g >= 0, 12.0, 3.0))
    return np.stack(fields, axis=0)