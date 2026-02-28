import os
import numpy as np
import matplotlib.pyplot as plt

from src.hjb_solver import HJBParams, solve_hjb_policy_iteration


def main():
    os.makedirs("results/plots", exist_ok=True)

    base = dict(
        T=1.0,
        Xmax=8.0,
        Nt=1200,
        Nx=801,
        sigma=0.7,
        alpha=1.0,
        umax=10.0,
        kappa=1.0,
        qT=1.0,
    )

    # Risk-neutral
    p0 = HJBParams(**base, theta=0.0)
    t0, x0, V0, U0 = solve_hjb_policy_iteration(
        p0, max_iter=60, tol=1e-3, damping=0.25, verbose=True
    )

    # Risk-averse
    p1 = HJBParams(
        **base,
        theta=1.0,
        picard_max_iter=10,
        picard_tol=1e-2,
    )
    t1, x1, V1, U1 = solve_hjb_policy_iteration(
        p1, max_iter=60, tol=2e-3, damping=0.15, verbose=True
    )

    # sanity: grids should match
    assert np.allclose(x0, x1)
    x = x0

    # Plots: control at t=0
    plt.figure()
    plt.plot(x, U0[0], label="theta=0 (risk-neutral)")
    plt.plot(x, U1[0], label="theta=1 (risk-averse)")
    plt.title("Optimal control u*(0,x): risk-neutral vs risk-averse")
    plt.xlabel("x")
    plt.ylabel("u*(0,x)")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/plots/U0_theta_compare.png", dpi=200, bbox_inches="tight")

    # Plots: ratio u/x away from 0
    eps = 1e-6
    mask = np.abs(x) > eps
    r0 = U0[0][mask] / x[mask]
    r1 = U1[0][mask] / x[mask]

    plt.figure()
    plt.plot(x[mask], r0, label="theta=0")
    plt.plot(x[mask], r1, label="theta=1")
    plt.title("Feedback gain proxy: u*(0,x)/x")
    plt.xlabel("x")
    plt.ylabel("u*(0,x)/x")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/plots/U0_over_x_theta_compare.png", dpi=200, bbox_inches="tight")

    # Value function at t=0
    plt.figure()
    plt.plot(x, V0[0], label="theta=0")
    plt.plot(x, V1[0], label="theta=1")
    plt.title("Value function V(0,x): risk-neutral vs risk-averse")
    plt.xlabel("x")
    plt.ylabel("V(0,x)")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/plots/V0_theta_compare.png", dpi=200, bbox_inches="tight")

    print("\nSaved comparison plots to results/plots/:")
    print(" - U0_theta_compare.png")
    print(" - U0_over_x_theta_compare.png")
    print(" - V0_theta_compare.png")

    print("\nSummary stats:")
    print(f"theta=0: V0 min/max = {float(np.min(V0[0])):.6f}, {float(np.max(V0[0])):.6f}")
    print(f"theta=1: V0 min/max = {float(np.min(V1[0])):.6f}, {float(np.max(V1[0])):.6f}")
    print(f"theta=0: U0 min/max = {float(np.min(U0[0])):.6f}, {float(np.max(U0[0])):.6f}")
    print(f"theta=1: U0 min/max = {float(np.min(U1[0])):.6f}, {float(np.max(U1[0])):.6f}")


if __name__ == "__main__":
    main()