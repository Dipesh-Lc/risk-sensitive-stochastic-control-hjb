import numpy as np

def solve_riccati_backward(T: float, Nt: int, alpha: float, qT: float):
    """
    Solve p'(t) = p(t)^2/alpha - 1 backward in time with terminal p(T)=qT.
    Returns t-grid and p(t).
    """
    t = np.linspace(0.0, T, Nt + 1)
    dt = t[1] - t[0]

    p = np.zeros(Nt + 1)
    p[-1] = qT

    # Backward Euler (stable): p_{n-1} = p_n - dt * (p_{n-1}^2/alpha - 1)
    # This is nonlinear in p_{n-1}; solve quadratic each step.
    for n in range(Nt, 0, -1):
        pn = p[n]
        # Solve for p_prev in: p_prev = pn - dt*(p_prev^2/alpha - 1)
        # => (dt/alpha) p_prev^2 + p_prev - (pn + dt) = 0
        A = dt / alpha
        B = 1.0
        C = -(pn + dt)

        disc = B*B - 4*A*C
        disc = max(disc, 0.0)
    
        p_prev = (-B + np.sqrt(disc)) / (2*A) if A > 0 else (pn + dt)

        p[n - 1] = p_prev

    return t, p


def riccati_control(x: np.ndarray, p_t: float, alpha: float):
    return -(p_t / alpha) * x