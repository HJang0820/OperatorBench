from pathlib import Path

import numpy as np
import matplotlib

# Codespaces에서 그래픽 창 없이 그림을 파일로 저장
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from operatorbench.data_generation.fields import (
    sample_fno_darcy_coefficients,
)
from operatorbench.data_generation.darcy_fd import solve_darcy_fd


def main():
    # 1. 이번 검사에 사용할 설정
    N = 85
    seed = 2022
    f_value = 1.0

    # 2. 계수장 한 개와 대응하는 수치해 생성
    a = sample_fno_darcy_coefficients(
        n_samples=1,
        N=N,
        seed=seed,
    )[0]

    u = solve_darcy_fd(a, f_value=f_value)

    # 3. 기본 검사
    assert a.shape == (N, N)
    assert u.shape == a.shape
    assert np.isfinite(a).all()
    assert np.isfinite(u).all()
    assert np.isin(a, [3.0, 12.0]).all()

    # 네 경계의 압력값이 0인지 확인
    assert np.all(u[[0, -1], :] == 0)
    assert np.all(u[:, [0, -1]] == 0)

    # 4. 기존 산술평균 5-point 차분식의 잔차 검사
    h = 1 / (N - 1)
    ac = a[1:-1, 1:-1]

    aE = 0.5 * (ac + a[2:, 1:-1])
    aW = 0.5 * (ac + a[:-2, 1:-1])
    aN = 0.5 * (ac + a[1:-1, 2:])
    aS = 0.5 * (ac + a[1:-1, :-2])

    Au = (
        (aE + aW + aN + aS) * u[1:-1, 1:-1]
        - aE * u[2:, 1:-1]
        - aW * u[:-2, 1:-1]
        - aN * u[1:-1, 2:]
        - aS * u[1:-1, :-2]
    ) / h**2

    rhs = np.full_like(Au, f_value)
    relative_residual = (
        np.linalg.norm(Au - rhs) / np.linalg.norm(rhs)
    )

    assert relative_residual < 1e-8

    # 5. 실제 격자 좌표 준비
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)

    # 6. 계수장과 수치해 시각화
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9, 4),
        constrained_layout=True,
    )

    # 현재 배열은 a[i, j] = a(x_i, y_j)로 해석한다.
    # 그림에서는 행이 y, 열이 x에 대응하므로 전치한다.
    im_a = axes[0].pcolormesh(
        x, y, a.T,
        shading="nearest",
        cmap="viridis",
        vmin=3,
        vmax=12,
    )

    im_u = axes[1].pcolormesh(
        x, y, u.T,
        shading="nearest",
        cmap="magma",
    )

    axes[0].set_title("Coefficient a(x, y)")
    axes[1].set_title("FDM reference u(x, y)")

    fig.colorbar(im_a, ax=axes[0], ticks=[3, 12])
    fig.colorbar(im_u, ax=axes[1])

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

    fig.suptitle(
        f"Darcy check | N={N}, seed={seed}, "
        f"f={f_value:g} | arithmetic faces"
    )

    # 7. 프로젝트 내부에 그림 저장
    project_root = Path(__file__).resolve().parents[1]
    figure_dir = project_root / "docs" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    figure_path = figure_dir / f"darcy_sample_N{N}_seed{seed}.png"

    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    # 8. 검사 결과 출력
    print("shape:", u.shape)
    print("u min/max:", u.min(), u.max())
    print("relative discrete residual:", relative_residual)
    print("Basic solver checks passed.")
    print("Saved figure:", figure_path)


if __name__ == "__main__":
    main()