import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def solve_darcy_fd(a, f_value=1):
    # 1. 입력 계수장을 float64 배열로 변환
    a = np.asarray(a, dtype=np.float64)

    # 경계를 포함하는 N×N 정사각형 격자를 가정
    N = a.shape[0]
    h = 1 / (N - 1)

    # 경계값은 이미 알고 있으므로 내부점만 미지수로 사용
    n_int = N - 2
    n_unknowns = n_int * n_int

    # 2. 희소행렬 원소와 우변 준비
    rows, cols, vals = [], [], []
    # Accept either a scalar or a full-grid source array.
    # rhs = np.full(n_unknowns, f_value, dtype=np.float64)
    f_grid = np.asarray(f_value, dtype=np.float64)

    if f_grid.ndim == 0:
        # Example: 1.0 becomes an (N, N) array filled with ones.
        f_grid = np.full(
            a.shape, 
            f_grid.item(),
            dtype=np.float64,
        )

    elif f_grid.shape != a.shape:
        raise ValueError(
            f"f_value must be a scalar or have shape {a.shape}; "
            f"received shape {f_grid.shape}."
        )

    if not np.isfinite(f_grid).all():
        raise ValueError(
            "f_value must contain only finite values."
        )
    # Use the source values at interior grid points.
    rhs = f_grid[1:-1, 1:-1].reshape(-1, order="C").copy()
    
    # 내부 격자의 (i, j)를 1차원 미지수 번호로 변환
    def idx(i, j):
        return (i - 1) * n_int + (j - 1)

    # 3. 각 내부점의 5-point stencil을 행렬에 기록
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            p = idx(i, j)

            # 기존 구현의 산술평균을 유지
            aE = 0.5 * (a[i, j] + a[i + 1, j])
            aW = 0.5 * (a[i, j] + a[i - 1, j])
            aN = 0.5 * (a[i, j] + a[i, j + 1])
            aS = 0.5 * (a[i, j] + a[i, j - 1])

            diag = (aE + aW + aN + aS) / h**2
            rows.append(p)
            cols.append(p)
            vals.append(diag)

            if i + 1 <= N - 2:
                rows.append(p)
                cols.append(idx(i + 1, j))
                vals.append(-aE / h**2)

            if i - 1 >= 1:
                rows.append(p)
                cols.append(idx(i - 1, j))
                vals.append(-aW / h**2)

            if j + 1 <= N - 2:
                rows.append(p)
                cols.append(idx(i, j + 1))
                vals.append(-aN / h**2)

            if j - 1 >= 1:
                rows.append(p)
                cols.append(idx(i, j - 1))
                vals.append(-aS / h**2)

    # 4. 희소행렬을 만들고 선형계 A u = rhs 풀이
    A_mat = sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(n_unknowns, n_unknowns),
    )
    u_int = spla.spsolve(A_mat, rhs)

    # 5. 내부 해를 N×N 배열에 배치하고 경계값 0 유지
    u = np.zeros((N, N), dtype=np.float64)

    for i in range(1, N - 1):
        for j in range(1, N - 1):
            u[i, j] = u_int[idx(i, j)]

    return u