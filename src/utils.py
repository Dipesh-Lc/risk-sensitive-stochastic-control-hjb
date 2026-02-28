import numpy as np

def make_grid(T: float, Xmax: float, Nt: int, Nx: int):
    t = np.linspace(0.0, T, Nt + 1)
    x = np.linspace(-Xmax, Xmax, Nx)
    dt = t[1] - t[0]
    dx = x[1] - x[0]
    return t, x, dt, dx

def enforce_neumann_bc(arr: np.ndarray):
    # Neumann BC: V_x = 0 => copy neighboring values
    arr[0] = arr[1]
    arr[-1] = arr[-2]
    return arr

def first_derivative_central(V: np.ndarray, dx: float):
    Vx = np.zeros_like(V)
    Vx[1:-1] = (V[2:] - V[:-2]) / (2.0 * dx)
    # one-sided at boundaries
    Vx[0] = (V[1] - V[0]) / dx
    Vx[-1] = (V[-1] - V[-2]) / dx
    return Vx

def second_derivative_central(V: np.ndarray, dx: float):
    Vxx = np.zeros_like(V)
    Vxx[1:-1] = (V[2:] - 2.0 * V[1:-1] + V[:-2]) / (dx * dx)
    # simple boundary handling (consistent with Neumann enforcement)
    Vxx[0] = Vxx[1]
    Vxx[-1] = Vxx[-2]
    return Vxx

def first_derivative_upwind(V: np.ndarray, dx: float, u: np.ndarray):
    """
    Upwind derivative based on sign of u:
      if u>0 use backward diff
      if u<0 use forward diff
    """
    dV = np.zeros_like(V)

    # backward diff
    back = (V[1:-1] - V[:-2]) / dx
    # forward diff
    fwd = (V[2:] - V[1:-1]) / dx

    mask = u[1:-1] >= 0
    dV[1:-1][mask] = back[mask]
    dV[1:-1][~mask] = fwd[~mask]

    # boundaries: one-sided
    dV[0] = (V[1] - V[0]) / dx
    dV[-1] = (V[-1] - V[-2]) / dx
    return dV