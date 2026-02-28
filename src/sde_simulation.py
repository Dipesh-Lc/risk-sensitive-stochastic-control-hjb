import numpy as np


def interp_control_bilinear(
    t_grid: np.ndarray,
    x_grid: np.ndarray,
    U_grid: np.ndarray,
    t: float,
    X: np.ndarray,
):
    """
    Bilinear interpolation of control u(t, x) from a grid U_grid[ti, xi].

    Parameters
    ----------
    t_grid : (Nt+1,)
    x_grid : (Nx,)
    U_grid : (Nt+1, Nx)
    t : scalar time
    X : (M,) state vector

    Returns
    -------
    u : (M,) interpolated control values
    """
    Nt = len(t_grid) - 1
    Nx = len(x_grid)

    # Clamp t into [t0, tT]
    t0, tT = t_grid[0], t_grid[-1]
    t_clamped = np.clip(t, t0, tT)

    # Find time cell index
    dt = t_grid[1] - t_grid[0]
    i = int(np.floor((t_clamped - t0) / dt))
    i = min(max(i, 0), Nt - 1)

    t_i = t_grid[i]
    wt = (t_clamped - t_i) / dt  # in [0,1]

    # Clamp X into [xmin, xmax]
    xmin, xmax = x_grid[0], x_grid[-1]
    Xc = np.clip(X, xmin, xmax)

    dx = x_grid[1] - x_grid[0]
    j = np.floor((Xc - xmin) / dx).astype(int)
    j = np.clip(j, 0, Nx - 2)

    x_j = x_grid[j]
    wx = (Xc - x_j) / dx

    # Gather four corners
    u00 = U_grid[i, j]
    u01 = U_grid[i, j + 1]
    u10 = U_grid[i + 1, j]
    u11 = U_grid[i + 1, j + 1]

    # Interpolate in x then t
    ux0 = (1.0 - wx) * u00 + wx * u01
    ux1 = (1.0 - wx) * u10 + wx * u11
    u = (1.0 - wt) * ux0 + wt * ux1

    return u


def simulate_controlled_sde_em(
    *,
    x0: float,
    T: float,
    Nt: int,
    sigma: float,
    control_fn,
    M: int,
    seed: int = 123,
):
    """
    Vectorized Euler-Maruyama simulation for:
        dX_t = u(t, X_t) dt + sigma dW_t

    Parameters
    ----------
    x0 : initial state
    T : terminal time
    Nt : number of time steps
    sigma : volatility
    control_fn : function (t, X)->u, where X is shape (M,)
    M : number of Monte Carlo paths
    seed : RNG seed

    Returns
    -------
    t : (Nt+1,)
    X : (Nt+1, M) simulated paths
    U : (Nt, M) applied controls at each step (t_n, X_n)
    """
    rng = np.random.default_rng(seed)
    dt = T / Nt
    sqrt_dt = np.sqrt(dt)

    t = np.linspace(0.0, T, Nt + 1)
    X = np.empty((Nt + 1, M), dtype=float)
    U = np.empty((Nt, M), dtype=float)

    X[0, :] = x0

    for n in range(Nt):
        tn = t[n]
        Xn = X[n, :]

        un = control_fn(tn, Xn)  # shape (M,)
        U[n, :] = un

        dW = rng.standard_normal(M) * sqrt_dt
        X[n + 1, :] = Xn + un * dt + sigma * dW

    return t, X, U


def simulate_controlled_sde_em_with_Z(
    *,
    x0: float,
    T: float,
    Nt: int,
    sigma: float,
    control_fn,
    Z: np.ndarray,  # shape (Nt, M), standard normal
):
    """
    Euler-Maruyama using pre-generated standard normal Z increments:
        X_{n+1} = X_n + u(t_n, X_n) dt + sigma * sqrt(dt) * Z_n

    Returns X (Nt+1, M) and U (Nt, M).
    """
    Nt_Z, M = Z.shape
    assert Nt_Z == Nt, "Z must have shape (Nt, M)"

    dt = T / Nt
    sqrt_dt = np.sqrt(dt)

    t = np.linspace(0.0, T, Nt + 1)
    X = np.empty((Nt + 1, M), dtype=float)
    U = np.empty((Nt, M), dtype=float)

    X[0, :] = x0

    for n in range(Nt):
        tn = t[n]
        Xn = X[n, :]
        un = control_fn(tn, Xn)
        U[n, :] = un
        X[n + 1, :] = Xn + un * dt + sigma * sqrt_dt * Z[n, :]

    return t, X, U