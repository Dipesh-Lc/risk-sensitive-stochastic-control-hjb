import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.hjb_solver import HJBParams, solve_hjb_policy_iteration
from src.sde_simulation import interp_control_bilinear, simulate_controlled_sde_em_with_Z
from src.monte_carlo_validation import estimate_risk_neutral_cost, estimate_risk_sensitive_cost_logexp


def feedback_gain_proxy(x, u0, window=1.0):
    mask = (np.abs(x) <= window) & (np.abs(x) > 1e-8)
    return float(np.mean(u0[mask] / x[mask]))


def solve_policy(theta, sigma, base, verbose=False):
    p = HJBParams(
        **base,
        sigma=float(sigma),
        theta=float(theta),
        picard_max_iter=10 if theta > 0 else 1,
        picard_tol=1e-2,
    )
    t, x, V, U = solve_hjb_policy_iteration(
        p,
        max_iter=60,
        tol=2e-3 if theta > 0 else 1e-3,
        damping=0.15 if theta > 0 else 0.25,
        verbose=verbose,
    )
    return p, t, x, V, U


def run_mc_for_policy(t, x, U, *, sigma, alpha, qT, T, M, Nt_mc, seed):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((Nt_mc, M))
    dt = T / Nt_mc

    def u_fn(tq, Xq):
        return interp_control_bilinear(t, x, U, tq, Xq)

    _, Xsim, Usim = simulate_controlled_sde_em_with_Z(
        x0=1.0, T=T, Nt=Nt_mc, sigma=sigma, control_fn=u_fn, Z=Z
    )

    J_mean, _, J_se = estimate_risk_neutral_cost(X=Xsim, U=Usim, dt=dt, alpha=alpha, qT=qT)
    return J_mean, J_se, Xsim, Usim, dt


def main():
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    base = dict(
        T=1.0, Xmax=8.0, Nt=800, Nx=601,
        alpha=1.0, umax=10.0, kappa=1.0, qT=1.0,
    )

    sigmas = [0.5, 0.7, 0.9, 1.1]
    thetas = [0.0, 0.25, 0.5, 0.75, 1.0]

    M = 15000
    Nt_mc = 1200
    seed = 123

    rows = []
    for sigma in sigmas:
        for theta in thetas:
            print(f"Solving PDE: sigma={sigma}, theta={theta} ...")
            p, t, x, V, U = solve_policy(theta=theta, sigma=sigma, base=base, verbose=False)

            gain = feedback_gain_proxy(x, U[0], window=1.0)

            # Monte Carlo (CRN within each (sigma,theta) run)
            J_mean, J_se, Xsim, Usim, dt = run_mc_for_policy(
                t, x, U,
                sigma=sigma, alpha=base["alpha"], qT=base["qT"],
                T=base["T"], M=M, Nt_mc=Nt_mc, seed=seed
            )

            theta_eval = float(theta) if theta > 0 else 1.0
            J_exp, J_exp_se = estimate_risk_sensitive_cost_logexp(
                X=Xsim, U=Usim, dt=dt, alpha=base["alpha"], qT=base["qT"], theta=theta_eval
            )

            rows.append(dict(
                sigma=float(sigma),
                theta=float(theta),
                gain_u_over_x=gain,
                J_mean=J_mean,
                J_mean_se=J_se,
                theta_eval=theta_eval,
                J_exp=J_exp,
                J_exp_se=J_exp_se,
            ))

    df = pd.DataFrame(rows)
    out_csv = "results/tables/sigma_theta_sensitivity.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    # Plots 
    # gain vs theta
    plt.figure()
    for sigma in sigmas:
        d = df[df["sigma"] == sigma].sort_values("theta")
        plt.plot(d["theta"], d["gain_u_over_x"], marker="o", label=f"sigma={sigma}")
    plt.title("Feedback gain proxy vs theta (by sigma)")
    plt.xlabel("theta")
    plt.ylabel("mean(u*(0,x)/x) for |x|<=1")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/plots/gain_vs_theta_by_sigma.png", dpi=200, bbox_inches="tight")

    # Jexp vs theta
    plt.figure()
    for sigma in sigmas:
        d = df[df["sigma"] == sigma].sort_values("theta")
        plt.errorbar(d["theta"], d["J_exp"], yerr=d["J_exp_se"], marker="o", capsize=3, label=f"sigma={sigma}")
    plt.title("Risk-sensitive exponential cost vs theta (by sigma)")
    plt.xlabel("theta")
    plt.ylabel("J_exp (mean ± approx SE)")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/plots/Jexp_vs_theta_by_sigma.png", dpi=200, bbox_inches="tight")

    # Jmean vs theta
    plt.figure()
    for sigma in sigmas:
        d = df[df["sigma"] == sigma].sort_values("theta")
        plt.errorbar(d["theta"], d["J_mean"], yerr=d["J_mean_se"], marker="o", capsize=3, label=f"sigma={sigma}")
    plt.title("Risk-neutral mean cost vs theta (by sigma)")
    plt.xlabel("theta")
    plt.ylabel("J_mean (mean ± SE)")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/plots/Jmean_vs_theta_by_sigma.png", dpi=200, bbox_inches="tight")

    print("Saved plots to results/plots/:")
    print(" - gain_vs_theta_by_sigma.png")
    print(" - Jexp_vs_theta_by_sigma.png")
    print(" - Jmean_vs_theta_by_sigma.png")


if __name__ == "__main__":
    main()