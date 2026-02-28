import os
import numpy as np
import matplotlib.pyplot as plt

from src.hjb_solver import HJBParams, solve_hjb_policy_iteration
from src.sde_simulation import interp_control_bilinear, simulate_controlled_sde_em_with_Z
from src.monte_carlo_validation import (
    estimate_risk_neutral_cost,
    estimate_risk_sensitive_cost_logexp,
)

def main():
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    # Fast grid config  
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

    # Solve theta=0
    p0 = HJBParams(**base, theta=0.0)
    t0, x0, V0, U0 = solve_hjb_policy_iteration(p0, max_iter=60, tol=1e-3, damping=0.25, verbose=True)

    # Solve theta=1 (risk-sensitive)
    p1 = HJBParams(**base, theta=1.0, picard_max_iter=10, picard_tol=1e-2)
    t1, x1, V1, U1 = solve_hjb_policy_iteration(p1, max_iter=60, tol=2e-3, damping=0.15, verbose=True)

    # Control functions
    def u_theta0(t, X):
        return interp_control_bilinear(t0, x0, U0, t, X)

    def u_theta1(t, X):
        return interp_control_bilinear(t1, x1, U1, t, X)

    # Monte Carlo settings 
    M = 50000
    x_init = 1.0
    Nt_mc = 2000
    theta_eval = 1.0  # evaluate risk-sensitive objective at this theta
    seed = 123

    dt = base["T"] / Nt_mc
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((Nt_mc, M))  # common random numbers

    # Simulate under both controls with same noise
    _, X_a, U_a = simulate_controlled_sde_em_with_Z(x0=x_init, T=base["T"], Nt=Nt_mc, sigma=base["sigma"], control_fn=u_theta0, Z=Z)
    _, X_b, U_b = simulate_controlled_sde_em_with_Z(x0=x_init, T=base["T"], Nt=Nt_mc, sigma=base["sigma"], control_fn=u_theta1, Z=Z)

    # Risk-neutral mean cost
    J0_mean, _, J0_se = estimate_risk_neutral_cost(X=X_a, U=U_a, dt=dt, alpha=base["alpha"], qT=base["qT"])
    J1_mean, _, J1_se = estimate_risk_neutral_cost(X=X_b, U=U_b, dt=dt, alpha=base["alpha"], qT=base["qT"])

    # Risk-sensitive exponential cost
    K0, K0_se = estimate_risk_sensitive_cost_logexp(X=X_a, U=U_a, dt=dt, alpha=base["alpha"], qT=base["qT"], theta=theta_eval)
    K1, K1_se = estimate_risk_sensitive_cost_logexp(X=X_b, U=U_b, dt=dt, alpha=base["alpha"], qT=base["qT"], theta=theta_eval)

    print("\nRisk-neutral mean cost E[∫ l dt + qT X_T^2]")
    print(f"policy theta=0: J={J0_mean:.6f} (SE={J0_se:.6f})")
    print(f"policy theta=1: J={J1_mean:.6f} (SE={J1_se:.6f})")

    print(f"\nRisk-sensitive exponential cost J_theta (theta_eval={theta_eval})")
    print(f"policy theta=0: Jθ={K0:.6f} (SE~{K0_se:.6f})")
    print(f"policy theta=1: Jθ={K1:.6f} (SE~{K1_se:.6f})")

    # Save summary
    out_path = "results/tables/mc_risk_sensitive_summary.txt"
    with open(out_path, "w") as f:
        f.write(f"Risk-neutral mean cost:\n")
        f.write(f"theta0_policy: {J0_mean:.10f} (SE={J0_se:.10f})\n")
        f.write(f"theta1_policy: {J1_mean:.10f} (SE={J1_se:.10f})\n\n")
        f.write(f"Risk-sensitive exponential cost (theta_eval={theta_eval}):\n")
        f.write(f"theta0_policy: {K0:.10f} (SE~{K0_se:.10f})\n")
        f.write(f"theta1_policy: {K1:.10f} (SE~{K1_se:.10f})\n")

    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()