import numpy as np
from dataclasses import dataclass

from src.utils import make_grid, first_derivative_upwind
from src.linalg import solve_tridiagonal


@dataclass
class HJBParams:
    # Domain / grid
    T: float = 1.0
    Xmax: float = 8.0
    Nt: int = 1200
    Nx: int = 801

    # Model params
    sigma: float = 0.7
    alpha: float = 1.0

    # Risk parameter (theta=0 risk-neutral; theta>0 risk-averse)
    theta: float = 0.0

    # Numerical stabilization
    umax: float = 10.0
    kappa: float = 1.0

    # Terminal cost: V(T,x) = qT * x^2
    qT: float = 1.0

    # Picard settings for risk term (used when theta>0)
    picard_max_iter: int = 10
    picard_tol: float = 2e-3


def _build_implicit_diffusion_tridiag(Nx: int, r: float):
    a = np.zeros(Nx, dtype=float)
    b = np.ones(Nx, dtype=float)
    c = np.zeros(Nx, dtype=float)

    # interior
    a[1:] = -r
    b[:] = 1.0 + 2.0 * r
    c[:-1] = -r

    # Dirichlet rows
    a[0] = 0.0
    b[0] = 1.0
    c[0] = 0.0

    a[-1] = 0.0
    b[-1] = 1.0
    c[-1] = 0.0

    return a, b, c


def _central_derivative(Vn: np.ndarray, dx: float):
    Vx = np.zeros_like(Vn)
    Vx[1:-1] = (Vn[2:] - Vn[:-2]) / (2.0 * dx)
    Vx[0] = (Vn[1] - Vn[0]) / dx
    Vx[-1] = (Vn[-1] - Vn[-2]) / dx
    return Vx


def _evaluate_policy_risk_sensitive(params: HJBParams, U_fixed: np.ndarray):
    """
    Policy evaluation for risk-sensitive HJB using Picard iteration on the nonlinear term:
      V_t + x^2 + alpha u^2 + u V_x + (sigma^2/2) V_xx + (theta/2) sigma^2 (V_x)^2 = 0

    We freeze (V_x)^2 using Vx_old from previous Picard iteration:
      risk_term = (theta/2) sigma^2 (Vx_old)^2
    treated as an additional running cost.

    Scheme per Picard iteration:
      implicit diffusion + explicit drift + explicit running_cost + explicit risk_term
    """
    t, x, dt, dx = make_grid(params.T, params.Xmax, params.Nt, params.Nx)

    # boundary values
    V_left = params.kappa * (x[0] ** 2)
    V_right = params.kappa * (x[-1] ** 2)

    r = 0.5 * (params.sigma ** 2) * dt / (dx * dx)
    a, b, c = _build_implicit_diffusion_tridiag(params.Nx, r)

    # Initial guess for V: use risk-neutral evaluation (risk_term = 0)
    V = np.zeros((params.Nt + 1, params.Nx), dtype=float)
    V[-1, :] = params.qT * (x**2)

    # Picard loop
    for it in range(params.picard_max_iter):
        V_old = V.copy()

        # compute frozen gradient
        Vx_old_grid = np.zeros_like(V_old)
        for n in range(params.Nt + 1):
            Vx_old_grid[n, :] = _central_derivative(V_old[n, :], dx)

        # compute NEW solution into separate array
        V_new = np.zeros_like(V_old)
        V_new[-1, :] = params.qT * (x**2)

        for n in range(params.Nt, 0, -1):
            Vn = V_new[n].copy()
            u = U_fixed[n].copy()

            Vx_up = first_derivative_upwind(Vn, dx, u)

            running_cost = x**2 + params.alpha * (u**2)
            drift_term = u * Vx_up

            #  CLIP gradient before squaring
            Vx_safe = np.clip(Vx_old_grid[n, :], -50.0, 50.0)
            risk_term = 0.5 * params.theta * (params.sigma**2) * (Vx_safe**2)

            rhs = Vn + dt * (running_cost + drift_term + risk_term)
            rhs[0] = V_left
            rhs[-1] = V_right

            V_new[n - 1, :] = solve_tridiagonal(a, b, c, rhs)

        #  Picard damping (critical)
        relax = 0.3
        V = (1.0 - relax) * V_old + relax * V_new

        # convergence check
        denom = np.max(np.abs(V_old)) + 1e-12
        diff = np.max(np.abs(V - V_old)) / denom

        if diff < params.picard_tol:
            break
    return t, x, dt, dx, V


def _evaluate_policy_risk_neutral(params: HJBParams, U_fixed: np.ndarray):
    """
    Risk-neutral evaluation (theta=0):
      V_t + x^2 + alpha u^2 + u V_x + (sigma^2/2) V_xx = 0
    """
    t, x, dt, dx = make_grid(params.T, params.Xmax, params.Nt, params.Nx)

    V = np.zeros((params.Nt + 1, params.Nx), dtype=float)
    V[-1, :] = params.qT * (x**2)

    V_left = params.kappa * (x[0] ** 2)
    V_right = params.kappa * (x[-1] ** 2)

    r = 0.5 * (params.sigma ** 2) * dt / (dx * dx)
    a, b, c = _build_implicit_diffusion_tridiag(params.Nx, r)

    for n in range(params.Nt, 0, -1):
        Vn = V[n].copy()
        u = U_fixed[n].copy()

        Vx_up = first_derivative_upwind(Vn, dx, u)

        running_cost = x**2 + params.alpha * (u**2)
        drift_term = u * Vx_up

        rhs = Vn + dt * (running_cost + drift_term)
        rhs[0] = V_left
        rhs[-1] = V_right

        V[n - 1, :] = solve_tridiagonal(a, b, c, rhs)

    return t, x, dt, dx, V


def _evaluate_policy(params: HJBParams, U_fixed: np.ndarray):
    if params.theta == 0.0:
        return _evaluate_policy_risk_neutral(params, U_fixed)
    return _evaluate_policy_risk_sensitive(params, U_fixed)


def _improve_policy(params: HJBParams, V: np.ndarray, dx: float):
    """
    Policy improvement remains:
      u*(t,x) = clip( -V_x / (2*alpha), [-umax, umax] )
    """
    Nt = V.shape[0] - 1
    U_new = np.zeros_like(V)

    for n in range(Nt + 1):
        Vx = _central_derivative(V[n], dx)
        u = -Vx / (2.0 * params.alpha)
        U_new[n] = np.clip(u, -params.umax, params.umax)

    return U_new


def solve_hjb_policy_iteration(
    params: HJBParams,
    max_iter: int = 60,
    tol: float = 1e-3,
    damping: float = 0.25,
    verbose: bool = True,
):
    """
    Policy iteration for both risk-neutral and risk-sensitive cases.
    For theta>0, each policy evaluation uses a Picard inner loop to handle (V_x)^2 term.
    """
    U = np.zeros((params.Nt + 1, params.Nx), dtype=float)

    last_diff = None
    for k in range(max_iter):
        t, x, dt, dx, V = _evaluate_policy(params, U)
        U_new = _improve_policy(params, V, dx)

        diff = float(np.max(np.abs(U_new - U)))
        last_diff = diff

        if verbose:
            print(f"[Policy Iter] iter={k:02d}  max|ΔU|={diff:.4e}")

        if diff < tol:
            U = U_new
            break

        U = (1.0 - damping) * U + damping * U_new

    if verbose and last_diff is not None:
        print(f"[Policy Iter] finished. last max|ΔU|={last_diff:.4e}")

    return t, x, V, U