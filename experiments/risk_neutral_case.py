import os
import numpy as np
import matplotlib.pyplot as plt

from src.hjb_solver import HJBParams, solve_hjb_policy_iteration
from src.riccati_benchmark import solve_riccati_backward, riccati_control

def main():
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    params = HJBParams(
        T=1.0,
        Xmax=8.0,
        Nt=1200,
        Nx=801,
        sigma=0.7,
        alpha=1.0,
        theta=0.0,
        umax=10.0,
        kappa=1.0,
        qT=1.0
    )

    t, x, V, U = solve_hjb_policy_iteration(
        params,
        max_iter=60,
        tol=1e-3,
        damping=0.25,
        verbose=True,
    )

    tR, p = solve_riccati_backward(params.T, params.Nt, params.alpha, params.qT)
    u_riccati_0 = riccati_control(x, p[0], params.alpha)

    # Error metrics 
    mask = (np.abs(x) <= 7.5)  # ignore very close boundaries
    err = U[0] - u_riccati_0

    l2 = np.sqrt(np.mean(err[mask]**2))
    linf = np.max(np.abs(err[mask]))
    rel_l2 = l2 / (np.sqrt(np.mean(u_riccati_0[mask]**2)) + 1e-12)

    print(f"L2 error (masked): {l2:.4e}")
    print(f"Linf error (masked): {linf:.4e}")
    print(f"Relative L2 error: {rel_l2:.4e}")

    with open("results/tables/risk_neutral_error.txt", "w") as f:
        f.write(f"L2_error_masked: {l2:.6e}\n")
        f.write(f"Linf_error_masked: {linf:.6e}\n")
        f.write(f"Rel_L2_error: {rel_l2:.6e}\n")
        

    # Save some diagnostics
    print("V0 min/max:", float(np.min(V[0])), float(np.max(V[0])))
    print("U0 min/max:", float(np.min(U[0])), float(np.max(U[0])))

    # Plot V(0,x)
    plt.figure()
    plt.plot(x, V[0])
    plt.title("Value function V(0,x) (risk-neutral θ=0)")
    plt.xlabel("x")
    plt.ylabel("V(0,x)")
    plt.grid(True)
    plt.savefig("results/plots/V0_risk_neutral.png", dpi=200, bbox_inches="tight")

    # Plot u*(0,x)
    plt.figure()
    plt.plot(x, U[0])
    plt.title("Optimal control u*(0,x) (risk-neutral θ=0)")
    plt.xlabel("x")
    plt.ylabel("u*(0,x)")
    plt.grid(True)
    plt.savefig("results/plots/U0_risk_neutral.png", dpi=200, bbox_inches="tight")

    # Extra sanity plot: U(0,x)/x (avoid x=0)
    eps = 1e-6
    mask = np.abs(x) > eps
    ratio = np.zeros_like(x)
    ratio[mask] = U[0][mask] / x[mask]

    plt.figure()
    plt.plot(x[mask], ratio[mask])
    plt.title("Sanity check: u*(0,x)/x (should be ~constant near 0)")
    plt.xlabel("x")
    plt.ylabel("u*(0,x)/x")
    plt.grid(True)
    plt.savefig("results/plots/U0_over_x_risk_neutral.png", dpi=200, bbox_inches="tight")

    # Compare PDE policy with Riccati benchmark
    plt.figure()
    plt.plot(x, U[0], label="PDE policy-iter u*(0,x)")
    plt.plot(x, u_riccati_0, "--", label="Riccati u*(0,x)")
    plt.title("Risk-neutral: PDE vs Riccati benchmark")
    plt.xlabel("x")
    plt.ylabel("u*(0,x)")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/plots/U0_vs_riccati.png", dpi=200, bbox_inches="tight")

    print("Saved plots to results/plots/")

if __name__ == "__main__":
    main()