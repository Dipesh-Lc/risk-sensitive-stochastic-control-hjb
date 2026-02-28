import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.hjb_solver import HJBParams, solve_hjb_policy_iteration
from src.sde_simulation import interp_control_bilinear, simulate_controlled_sde_em_with_Z
from src.monte_carlo_validation import (
    estimate_risk_neutral_cost,
    estimate_risk_sensitive_cost_logexp,
)

def feedback_gain_proxy(x, u0, window=1.0):
    """Estimate slope near 0 using average u/x over |x|<=window excluding x=0."""
    mask = (np.abs(x) <= window) & (np.abs(x) > 1e-8)
    return float(np.mean(u0[mask] / x[mask]))

def run_one(theta, sigma=0.7, M=50000, Nt_mc=2000, seed=123):
    # Fast grid settings
    base = dict(
        T=1.0, Xmax=8.0, Nt=1200, Nx=801,
        sigma=sigma, alpha=1.0,
        umax=10.0, kappa=1.0, qT=1.0,
    )

    p = HJBParams(
        **base,
        theta=float(theta),
        picard_max_iter=10 if theta > 0 else 1,
        picard_tol=1e-2,
    )

    t, x, V, U = solve_hjb_policy_iteration(
        p,
        max_iter=60,
        tol=2e-3 if theta > 0 else 1e-3,
        damping=0.15 if theta > 0 else 0.25,
        verbose=False,
    )

    # feedback proxy
    gain = feedback_gain_proxy(x, U[0], window=1.0)

    # MC with CRN (so comparisons across theta are less noisy per run)
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((Nt_mc, M))

    def u_policy(tq, Xq):
        return interp_control_bilinear(t, x, U, tq, Xq)

    _, Xsim, Usim = simulate_controlled_sde_em_with_Z(
        x0=1.0, T=base["T"], Nt=Nt_mc, sigma=sigma, control_fn=u_policy, Z=Z
    )

    dt = base["T"] / Nt_mc
    J_mean, _, J_se = estimate_risk_neutral_cost(X=Xsim, U=Usim, dt=dt, alpha=base["alpha"], qT=base["qT"])

    # Evaluate exponential cost at theta_eval = max(theta, small)
    theta_eval = float(theta) if theta > 0 else 1.0
    Jexp, Jexp_se = estimate_risk_sensitive_cost_logexp(
        X=Xsim, U=Usim, dt=dt, alpha=base["alpha"], qT=base["qT"], theta=theta_eval
    )

    return {
        "theta": float(theta),
        "sigma": float(sigma),
        "gain_u_over_x": gain,
        "J_mean": J_mean,
        "J_mean_se": J_se,
        "theta_eval": theta_eval,
        "J_exp": Jexp,
        "J_exp_se": Jexp_se,
    }

def main():
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    thetas = [0.0, 0.25, 0.5, 0.75, 1.0]
    sigmas = [0.7]

    rows = []
    for sigma in sigmas:
        for theta in thetas:
            print(f"Running theta={theta}, sigma={sigma} ...")
            rows.append(run_one(theta=theta, sigma=sigma, M=10000, Nt_mc=800, seed=123))

    df = pd.DataFrame(rows)
    out_csv = "results/tables/sensitivity_theta.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    # Plot gain vs theta
    plt.figure()
    for sigma in sigmas:
        d = df[df["sigma"] == sigma].sort_values("theta")
        plt.plot(d["theta"], d["gain_u_over_x"], marker="o", label=f"sigma={sigma}")
    plt.title("Feedback gain proxy vs theta (near x=0)")
    plt.xlabel("theta")
    plt.ylabel("mean(u*(0,x)/x) for |x|<=1")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/plots/gain_vs_theta.png", dpi=200, bbox_inches="tight")

    # Plot exponential cost vs theta (using theta_eval=theta except theta=0 uses 1.0)
    plt.figure()
    for sigma in sigmas:
        d = df[df["sigma"] == sigma].sort_values("theta")
        plt.errorbar(d["theta"], d["J_exp"], yerr=d["J_exp_se"], marker="o", capsize=3, label=f"sigma={sigma}")
    plt.title("Risk-sensitive exponential cost vs theta")
    plt.xlabel("theta")
    plt.ylabel("J_exp (mean ± approx SE)")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/plots/Jexp_vs_theta.png", dpi=200, bbox_inches="tight")

    print("Saved plots to results/plots/: gain_vs_theta.png, Jexp_vs_theta.png")

if __name__ == "__main__":
    main()