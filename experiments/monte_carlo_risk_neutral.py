import os
import numpy as np
import matplotlib.pyplot as plt

from src.hjb_solver import HJBParams, solve_hjb_policy_iteration
from src.riccati_benchmark import solve_riccati_backward
from src.sde_simulation import interp_control_bilinear
from src.monte_carlo_validation import run_mc_comparison
from src.monte_carlo_validation import run_mc_comparison_crn


def main():
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    # HJB solve params (grid for PDE control) 
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
        qT=1.0,
    )

    # Solve PDE via policy iteration
    t_grid, x_grid, V, U_grid = solve_hjb_policy_iteration(
        params, max_iter=60, tol=1e-3, damping=0.25, verbose=True
    )

    # Riccati benchmark p(t) for analytic control
    tR, p = solve_riccati_backward(params.T, params.Nt, params.alpha, params.qT)

    # Control functions for Monte Carlo (feedback)
    def pde_control_fn(t, X):
        return interp_control_bilinear(t_grid, x_grid, U_grid, t, X)

    def riccati_control_fn(t, X):
        # linear interpolation of p(t)
        dtR = tR[1] - tR[0]
        i = int(np.clip(np.floor(t / dtR), 0, len(p) - 2))
        w = (t - tR[i]) / dtR
        p_t = (1 - w) * p[i] + w * p[i + 1]
        return -(p_t / params.alpha) * X

    # Monte Carlo settings 
    M = 50000          
    x0 = 1.0
    Nt_mc = 2000       # MC time steps 
    seed = 123

    results = run_mc_comparison_crn(
        M=M,
        x0=x0,
        T=params.T,
        Nt_mc=Nt_mc,
        sigma=params.sigma,
        alpha=params.alpha,
        qT=params.qT,
        pde_control_fn=pde_control_fn,
        riccati_control_fn=riccati_control_fn,
        seed=seed,
    )

    # Print results
    print("\nMonte Carlo comparison (risk-neutral cost):")
    for k in ["pde", "riccati", "zero"]:
        v = results[k]
        print(f"{k:8s}: J={v['J_mean']:.6f}  (SE={v['J_se']:.6f})")

    d = results["pde_minus_riccati"]
    print(f"\nPaired diff (PDE - Riccati): mean={d['mean']:.6f}  (SE={d['se']:.6f})")
    
    # Save results
    out_path = "results/tables/mc_risk_neutral_summary.txt"
    with open(out_path, "w") as f:
        for k in ["pde", "riccati", "zero"]:
            v = results[k]
            f.write(f"{k}: J_mean={v['J_mean']:.10f}, J_se={v['J_se']:.10f}, J_std={v['J_std']:.10f}\n")

        d = results["pde_minus_riccati"]
        f.write(f"pde_minus_riccati: mean={d['mean']:.10f}, se={d['se']:.10f}, std={d['std']:.10f}\n")

    # Simple bar plot
    labels = ["zero", "riccati", "pde"]
    means = [results[l]["J_mean"] for l in labels]
    ses = [results[l]["J_se"] for l in labels]

    plt.figure()
    plt.bar(labels, means, yerr=ses)
    plt.title(f"Monte Carlo estimated cost (M={M}, x0={x0})")
    plt.ylabel("Estimated J (mean ± SE)")
    plt.grid(True, axis="y")
    plt.savefig("results/plots/mc_cost_bar.png", dpi=200, bbox_inches="tight")
    print("Saved plot: results/plots/mc_cost_bar.png")


if __name__ == "__main__":
    main()