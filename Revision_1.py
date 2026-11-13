# ============================================================
# FLOQUET STABILITY ANALYSIS FOR IMPULSIVE STEM-CELL THERAPY
# Manuscript MBE-8509
#
# Reviewer Comments 4, 5 and 7
#
# Comment 4:
#   Numerical Floquet multipliers for baseline dosing schedule
#
# Comment 5:
#   Local sensitivity of Floquet stability to xi and eta
#
# Comment 7:
#   One-parameter dosing/stability analysis with respect to b and tau
#
# Model: System (12) of the manuscript
# Baseline therapy: b = 6 cells/mm^3, tau = 10 days
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp


# ============================================================
# 1. MODEL PARAMETERS FROM TABLE 2
# ============================================================

PARAMS = {

    # Stem-cell parameters
    "rho0": 0.1,

    "b1": 0.55,
    "b2": 0.35,

    "l1": 0.47,
    "l2": 0.60,

    "lambda1": 0.125,
    "lambda2": 0.343,
    "lambdaM": 1.1,

    # Immune/keratinocyte parameters
    "CT": 15.0,
    "CD": 20.0,
    "CK": 30.0,

    "gamma": 0.001,
    "delta1": 0.0002,
    "delta2": 0.001,

    # MSC effects
    "xi": 0.007,
    "eta": 0.1,

    # Natural mortality
    "lambdaT": 0.07,
    "lambdaD": 0.15,
    "lambdaK": 0.2,
}


# ============================================================
# 2. DERIVED STEM-CELL PARAMETERS
#
# Manuscript:
#
# sigma_i = lambda_i - (2*b_i - 1)*l_i
# rho_i   = 2*(1-b_i)*l_i
#
# for i = 1,2
# ============================================================

def derived_parameters(p):

    p = p.copy()

    p["sigma1"] = (p["lambda1"] - (2.0 * p["b1"] - 1.0) * p["l1"])

    p["sigma2"] = (p["lambda2"] - (2.0 * p["b2"] - 1.0) * p["l2"])

    p["rho1"] = (2.0 * (1.0 - p["b1"]) * p["l1"])

    p["rho2"] = (2.0 * (1.0 - p["b2"]) * p["l2"])

    return p


PARAMS = derived_parameters(PARAMS)


print("\nDerived parameters")
print("------------------")
print(f"sigma1 = {PARAMS['sigma1']:.6f}")
print(f"sigma2 = {PARAMS['sigma2']:.6f}")
print(f"rho1   = {PARAMS['rho1']:.6f}")
print(f"rho2   = {PARAMS['rho2']:.6f}")


# ============================================================
# 3. IMPULSIVE MODEL: SYSTEM (12)
#
# Between impulses:
#
# dS1/dt = rho0 - sigma1*S1
# dS2/dt = rho1*S1 - sigma2*S2
# dSM/dt = rho2*S2 - lambdaM*SM
# dT/dt = CT - gamma*T*D - delta1*T*K- xi*T*SM - lambdaT*T
#dD/dt = CD - delta2*D*K - lambdaD*D
#dK/dt = CK + (delta1*T*K + delta2*D*K)/(1 + eta*SM) - lambdaK*K
#
# Impulse:
#S1(n*tau+) = S1(n*tau-) + b
# ============================================================

def model_rhs(t, y, p):

    S1, S2, SM, T, D, K = y

    dS1 = (p["rho0"] - p["sigma1"] * S1)

    dS2 = (p["rho1"] * S1 - p["sigma2"] * S2)

    dSM = (p["rho2"] * S2 - p["lambdaM"] * SM)

    dT = (p["CT"] - p["gamma"] * T * D - p["delta1"] * T * K - p["xi"] * T * SM - p["lambdaT"] * T)

    dD = (p["CD"] - p["delta2"] * D * K - p["lambdaD"] * D)

    denominator = 1.0 + p["eta"] * SM

    dK = (p["CK"] + ( p["delta1"] * T * K + p["delta2"] * D * K) / denominator - p["lambdaK"] * K)

    return np.array([dS1, dS2, dSM, dT, dD, dK])


# ============================================================
# 4. JACOBIAN OF SYSTEM (12)
#
# This is the time-dependent Jacobian evaluated along the
# periodic orbit.
#
# This is what we use for the numerical variational equation:
#
# dPhi/dt = J(t) Phi
#
# ============================================================

def jacobian(y, p):

    S1, S2, SM, T, D, K = y

    eta = p["eta"]

    denominator = 1.0 + eta * SM

    denominator_sq = denominator ** 2

    J = np.zeros((6, 6))

    # --------------------------------------------------------
    # S1 equation
    # --------------------------------------------------------

    J[0, 0] = -p["sigma1"]


    # --------------------------------------------------------
    # S2 equation
    # --------------------------------------------------------

    J[1, 0] = p["rho1"]
    J[1, 1] = -p["sigma2"]


    # --------------------------------------------------------
    # SM equation
    # --------------------------------------------------------

    J[2, 1] = p["rho2"]
    J[2, 2] = -p["lambdaM"]


    # --------------------------------------------------------
    # T equation
    # --------------------------------------------------------

    J[3, 2] = -p["xi"] * T

    J[3, 3] = (-p["gamma"] * D -p["delta1"] * K -p["xi"] * SM -p["lambdaT"])

    J[3, 4] = -p["gamma"] * T

    J[3, 5] = -p["delta1"] * T


    # --------------------------------------------------------
    # D equation
    # --------------------------------------------------------

    J[4, 4] = (-p["delta2"] * K -p["lambdaD"])

    J[4, 5] = -p["delta2"] * D


    # --------------------------------------------------------
    # K equation
    # --------------------------------------------------------

    numerator = (p["delta1"] * T * K + p["delta2"] * D * K)

    # derivative with respect to SM
    J[5, 2] = (-eta * numerator/ denominator_sq)

    # derivative with respect to T
    J[5, 3] = (p["delta1"] * K/ denominator)

    # derivative with respect to D
    J[5, 4] = (p["delta2"] * K/ denominator)

    # derivative with respect to K
    J[5, 5] = ((p["delta1"] * T + p["delta2"] * D)/ denominator- p["lambdaK"])

    return J


# ============================================================
# 5. ONE IMPULSIVE CYCLE
#
# Starting point is immediately AFTER an impulse.
#
# Integrate for tau days.
# Then apply:
#
# S1+ = S1- + b
#
# ============================================================

def one_cycle(x_post, b, tau, p):

    sol = solve_ivp(fun=lambda t, y: model_rhs(t, y, p),t_span=(0.0, tau),y0=x_post,method="DOP853",rtol=1e-10,atol=1e-12)

    if not sol.success:raise RuntimeError( "ODE integration failed during one cycle.")

    x_pre = sol.y[:, -1].copy()

    # Impulse
    x_post_new = x_pre.copy()
    x_post_new[0] += b

    return x_post_new


# ============================================================
# 6. CONVERGE TO THE IMPULSIVE PERIODIC ORBIT
#
# We iterate the Poincare map:
#
# x_{n+1} = P(x_n)
#
# until successive post-impulse states become identical.
# ============================================================

def converge_to_periodic_orbit(x_initial,b,tau,p,max_cycles=500,tolerance=1e-10):

    x = np.array(x_initial, dtype=float)

    for cycle in range(1, max_cycles + 1):

        x_new = one_cycle(x,b,tau,p)

        error = np.max(np.abs(x_new - x))

        x = x_new

        if error < tolerance:

            return x, cycle

    print("Warning: periodic convergence tolerance was not reached.")

    return x, max_cycles


# ============================================================
# 7. NUMERICAL MONODROMY MATRIX
#
# Variational equation:
#
# dPhi/dt = J(t) Phi
#
# Phi(0) = I_6
#
# Because the impulse is additive:
#
# S1+ = S1- + b
#
# its derivative with respect to the state is the identity.
#
# Therefore the jump matrix is:
#
# Dg = I_6
#
# and the one-cycle monodromy matrix is simply Phi(tau).
# ============================================================

def compute_monodromy(x_post,b,tau,p):

    Phi0 = np.eye(6)

    z0 = np.concatenate([x_post, Phi0.flatten()])

    def augmented_rhs(t, z):

        x = z[:6]

        Phi = z[6:].reshape(6, 6)

        dx = model_rhs(t,x,p)

        J = jacobian(x,p)

        dPhi = J @ Phi

        return np.concatenate([dx,dPhi.flatten()])

    sol = solve_ivp(fun=augmented_rhs,t_span=(0.0, tau),y0=z0,method="DOP853",rtol=1e-10,atol=1e-12)

    if not sol.success:raise RuntimeError("Variational equation integration failed.")

    x_pre = sol.y[:6, -1].copy()

    Phi_tau = (sol.y[6:, -1].reshape(6, 6))

    # Apply the impulse to the state only.
    # The derivative of the additive jump is identity,
    # so the monodromy matrix is unchanged.
    x_post_new = x_pre.copy()
    x_post_new[0] += b

    return (x_pre,x_post_new,Phi_tau)


# ============================================================
# 8. FLOQUET MULTIPLIERS
# ============================================================

def calculate_floquet_multipliers(b, tau, p, x_initial):

    # --------------------------------------------------------
    # Step 1: converge to periodic orbit
    # --------------------------------------------------------

    x_post, cycles = converge_to_periodic_orbit(x_initial=x_initial, b=b, tau=tau, p=p)

    # --------------------------------------------------------
    # Step 2: integrate variational equation for one cycle
    # --------------------------------------------------------

    x_pre, x_post_check, M = compute_monodromy(x_post=x_post, b=b, tau=tau, p=p)

    # --------------------------------------------------------
    # Step 3: eigenvalues of full monodromy matrix
    # --------------------------------------------------------

    full_multipliers = np.linalg.eigvals(M)

    # --------------------------------------------------------
    # Step 4: R1 and R2 blocks
    #
    # Manuscript decomposes the monodromy matrix into
    # an upper-left 3x3 stem-cell block and a lower-right
    # 3x3 immune/keratinocyte block.
    # --------------------------------------------------------

    R1 = M[:3, :3]

    R2 = M[3:, 3:]

    R1_multipliers = np.linalg.eigvals(R1)

    R2_multipliers = np.linalg.eigvals(R2)

    # Spectral radius of R2
    rho_R2 = np.max(np.abs(R2_multipliers))

    # Spectral radius of complete monodromy matrix
    rho_full = np.max(np.abs(full_multipliers))

    return {
        "x_post": x_post,
        "x_pre": x_pre,
        "monodromy": M,
        "full_multipliers": full_multipliers,
        "R1_multipliers": R1_multipliers,
        "R2_multipliers": R2_multipliers,
        "rho_R2": rho_R2,
        "rho_full": rho_full,
        "cycles": cycles
    }


# ============================================================
# 9. INITIAL CONDITION
#
# Manuscript Figure 3 reports:
#
# E* = (1.28, 1.04, 0.74, 86.5, 53.5, 223.8)
#
# We use this as the initial state for the treatment
# dynamics and then converge numerically to the periodic orbit.
# ============================================================

x0 = np.array([
    1.28,    # S1
    1.04,    # S2
    0.74,    # SM
    86.5,    # T
    53.5,    # D
    223.8    # K
])


# ============================================================
# 10. BASELINE THERAPY
#
# Manuscript's effective treatment strategy:
#
# b   = 6 cells/mm^3
# tau = 10 days
# ============================================================

BASELINE_B = 6.0
BASELINE_TAU = 10.0


baseline = calculate_floquet_multipliers(
    b=BASELINE_B,
    tau=BASELINE_TAU,
    p=PARAMS,
    x_initial=x0
)


# ============================================================
# 11. PRINT BASELINE RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("BASELINE FLOQUET ANALYSIS")
print("=" * 70)

print(
    f"Baseline dose b   = {BASELINE_B:.2f} cells/mm^3"
)

print(
    f"Baseline interval tau = {BASELINE_TAU:.2f} days"
)

print(
    f"Periodic orbit reached after "
    f"{baseline['cycles']} cycles"
)


print("\nPost-impulse periodic state:")
print("--------------------------------")

labels = [
    "S1",
    "S2",
    "SM",
    "T",
    "D",
    "K"
]

for label, value in zip(
        labels,
        baseline["x_post"]):

    print(
        f"{label:>3s} = {value:.10f}"
    )


# ============================================================
# 12. CHECK PERIODICITY
# ============================================================

periodicity_error = np.max(
    np.abs(
        baseline["x_post"]
        -
        (
            baseline["x_pre"]
            + np.array([BASELINE_B, 0, 0, 0, 0, 0])
        )
    )
)

print(
    f"\nPeriodic-orbit closure error = "
    f"{periodicity_error:.3e}"
)


# ============================================================
# 13. PRINT ALL SIX FLOQUET MULTIPLIERS
# ============================================================

print("\n")
print("Six Floquet multipliers")
print("--------------------------------")

full_mult = baseline["full_multipliers"]

for i, mu in enumerate(full_mult, start=1):

    print(
        f"mu_{i} = "
        f"{mu.real:.10e}"
        f"{mu.imag:+.10e}j"
        f"   |mu| = {abs(mu):.10e}"
    )


# ============================================================
# 14. PRINT R1 MULTIPLIERS
# ============================================================

print("\n")
print("R1 Floquet multipliers")
print("--------------------------------")

for i, mu in enumerate(
        baseline["R1_multipliers"],
        start=1):

    print(
        f"R1_mu_{i} = "
        f"{mu.real:.10e}"
        f"{mu.imag:+.10e}j"
        f"   |mu| = {abs(mu):.10e}"
    )


# ============================================================
# 15. PRINT R2 MULTIPLIERS
#
# These are the important ones for Comment 5 because
# manuscript states that stability depends on R2.
# ============================================================

print("\n")
print("R2 Floquet multipliers")
print("--------------------------------")

for i, mu in enumerate(
        baseline["R2_multipliers"],
        start=1):

    print(
        f"R2_mu_{i} = "
        f"{mu.real:.10e}"
        f"{mu.imag:+.10e}j"
        f"   |mu| = {abs(mu):.10e}"
    )


print(
    f"\nSpectral radius of R2 = "
    f"{baseline['rho_R2']:.10e}"
)

print(
    f"Spectral radius of full monodromy matrix = "
    f"{baseline['rho_full']:.10e}"
)


if baseline["rho_R2"] < 1.0:

    print(
        "\nRESULT: The impulsive periodic orbit is "
        "asymptotically stable."
    )

else:

    print(
        "\nRESULT: The impulsive periodic orbit is "
        "unstable."
    )

# ============================================================
# 16. SAVE BASELINE FLOQUET MULTIPLIERS
# ============================================================

baseline_rows = []

for i, mu in enumerate(
        baseline["full_multipliers"],
        start=1):

    baseline_rows.append({

        "Multiplier": f"mu_{i}",

        "Real": mu.real,

        "Imaginary": mu.imag,

        "Absolute_value": abs(mu)
    })

baseline_df = pd.DataFrame(
    baseline_rows
)

baseline_df.to_csv("baseline_floquet_multipliers.csv", index=False)

# ============================================================
# 17. COMMENT 5
# LOCAL SENSITIVITY ANALYSIS OF xi AND eta
#
# We vary each parameter independently while keeping
# everything else fixed.
#
# Recommended range:
#
# 50%, 75%, 100%, 125%, 150%
#
# of baseline value.
#
# Stability metric:
#
# rho_R2 = max |mu_i^(R2)|
#
# ============================================================

sensitivity_factors = np.array([
    0.50,
    0.75,
    1.00,
    1.25,
    1.50
])

sensitivity_results = []

for parameter_name in ["xi", "eta"]:

    baseline_value = PARAMS[parameter_name]

    for factor in sensitivity_factors:

        p_local = PARAMS.copy()

        p_local[parameter_name] = (
            baseline_value * factor
        )

        result = calculate_floquet_multipliers(
            b=BASELINE_B,
            tau=BASELINE_TAU,
            p=p_local,
            x_initial=x0
        )

        rho = result["rho_R2"]

        sensitivity_results.append({

            "Parameter": parameter_name,

            "Multiplier_factor": factor,

            "Parameter_value":
                p_local[parameter_name],

            "rho_R2": rho,

            "Stable": rho < 1.0,

            "K_post_impulse":
                result["x_post"][5]
        })


sensitivity_df = pd.DataFrame(
    sensitivity_results
)


print("\n")
print("=" * 70)
print("LOCAL SENSITIVITY ANALYSIS: xi AND eta")
print("=" * 70)

print(
    sensitivity_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)


sensitivity_df.to_csv(
    "xi_eta_floquet_sensitivity.csv",
    index=False
)


# ============================================================

# ============================================================
# 18. COMMENT 5
# LOCAL SENSITIVITY ANALYSIS OF xi AND eta
#
# The parameters are varied independently over 50%, 75%, 100%,
# 125%, and 150% of their respective BASELINE values.
# The x-axis of the final plots uses the ACTUAL parameter values.
#
# Stability metric:
#     rho_R2 = max |mu_i^(R2)|
#
# Baseline values:
#     xi  = 0.007
#     eta = 0.1
# ============================================================

sensitivity_factors = np.array([
    0.50,
    0.75,
    1.00,
    1.25,
    1.50
])

sensitivity_results = []

for parameter_name in ["xi", "eta"]:

    baseline_value = PARAMS[parameter_name]

    for factor in sensitivity_factors:

        p_local = PARAMS.copy()
        p_local[parameter_name] = baseline_value * factor

        result = calculate_floquet_multipliers(
            b=BASELINE_B,
            tau=BASELINE_TAU,
            p=p_local,
            x_initial=x0
        )

        sensitivity_results.append({
            "Parameter": parameter_name,
            "Multiplier_factor": factor,
            "Parameter_value": p_local[parameter_name],
            "rho_R2": result["rho_R2"],
            "Stable": result["rho_R2"] < 1.0,
            "K_post_impulse": result["x_post"][5]
        })

sensitivity_df = pd.DataFrame(sensitivity_results)

print("\n")
print("=" * 70)
print("LOCAL SENSITIVITY ANALYSIS: xi AND eta")
print("=" * 70)

print(
    sensitivity_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)

sensitivity_df.to_csv(
    "xi_eta_floquet_sensitivity.csv",
    index=False
)


# ============================================================
# 19. ONE-PARAMETER ANALYSIS WITH RESPECT TO STEM-CELL DOSE b
# ============================================================

b_values = np.linspace(
    0.5,
    12.0,
    24
)

dose_results = []

for b_value in b_values:

    result = calculate_floquet_multipliers(
        b=b_value,
        tau=BASELINE_TAU,
        p=PARAMS,
        x_initial=x0
    )

    dose_results.append({
        "b": b_value,
        "rho_R2": result["rho_R2"],
        "K_post_impulse": result["x_post"][5],
        "Stable": result["rho_R2"] < 1.0
    })

dose_df = pd.DataFrame(dose_results)

dose_df.to_csv(
    "dose_b_floquet_analysis.csv",
    index=False
)


# ============================================================
# 20. ONE-PARAMETER ANALYSIS WITH RESPECT TO INFUSION INTERVAL tau
# ============================================================

tau_values = np.linspace(
    5.0,
    20.0,
    16
)

tau_results = []

for tau_value in tau_values:

    result = calculate_floquet_multipliers(
        b=BASELINE_B,
        tau=tau_value,
        p=PARAMS,
        x_initial=x0
    )

    tau_results.append({
        "tau": tau_value,
        "rho_R2": result["rho_R2"],
        "K_post_impulse": result["x_post"][5],
        "Stable": result["rho_R2"] < 1.0
    })

tau_df = pd.DataFrame(tau_results)

tau_df.to_csv(
    "tau_floquet_analysis.csv",
    index=False
)


# ============================================================
# 21. CHECK FLOQUET STABILITY TRANSITIONS
# ============================================================

def check_stability_transition(df, parameter_column):

    rho_values = df["rho_R2"].values
    stable_values = rho_values < 1.0

    changes = np.where(
        stable_values[:-1] != stable_values[1:]
    )[0]

    if len(changes) == 0:
        print(
            f"No Floquet stability transition "
            f"(rho_R2 = 1 crossing) was detected over the "
            f"examined {parameter_column} range."
        )
    else:
        print(
            f"Possible Floquet stability transition(s) detected "
            f"over the examined {parameter_column} range:"
        )

        for idx in changes:
            print(
                f"{parameter_column} = "
                f"{df.loc[idx, parameter_column]:.6f} "
                f"to "
                f"{df.loc[idx + 1, parameter_column]:.6f}"
            )


print("\nFloquet stability transition check for b")
print("-----------------------------------------")
check_stability_transition(dose_df, "b")

print("\nFloquet stability transition check for tau")
print("-------------------------------------------")
check_stability_transition(tau_df, "tau")


# ============================================================
# 22. HEALTHY KERATINOCYTE THRESHOLD
# ============================================================

K_THRESHOLD = 200.0

print("\nHealthy keratinocyte threshold")
print("--------------------------------")

b_below = dose_df[dose_df["K_post_impulse"] < K_THRESHOLD]

if len(b_below) > 0:
    print(
        f"First examined dose with K < {K_THRESHOLD:.1f}: "
        f"b = {b_below.iloc[0]['b']:.4f}"
    )
else:
    print(
        "No dose in the examined range produced "
        "K below the healthy threshold."
    )

tau_below = tau_df[tau_df["K_post_impulse"] < K_THRESHOLD]

if len(tau_below) > 0:
    print(
        f"First examined interval with K < {K_THRESHOLD:.1f}: "
        f"tau = {tau_below.iloc[0]['tau']:.4f} days"
    )
else:
    print(
        "No infusion interval in the examined range produced "
        "K below the healthy threshold."
    )


# ============================================================
# 23. FOUR-PANEL FIGURE: 2 x 2
#
# (a) xi sensitivity
# (b) eta sensitivity
# (c) dose b: Floquet multiplier + keratinocyte level
# (d) interval tau: Floquet multiplier + keratinocyte level
#
# For xi and eta, the x-axis shows the ACTUAL parameter values,
# not normalized factors.
#
# For b and tau, the keratinocyte level is shown in the same
# manner as the dose analysis.
# ============================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(14, 10)
)

ax_xi = axes[0, 0]
ax_eta = axes[0, 1]
ax_b = axes[1, 0]
ax_tau = axes[1, 1]


# ------------------------------------------------------------
# (a) xi sensitivity
# ------------------------------------------------------------

subset_xi = sensitivity_df[
    sensitivity_df["Parameter"] == "xi"
]

xi_baseline = PARAMS["xi"]

ax_xi.plot(
    subset_xi["Parameter_value"],
    subset_xi["rho_R2"],
    marker="o",
    linewidth=2,
    label=r"Floquet multiplier"
)

ax_xi.axhline(
    y=1.0,
    linestyle="--",
    linewidth=2,
    color = "orange",
    label=r"Stability boundary = 1"
)

ax_xi.axvline(
    x=xi_baseline,
    linestyle=":",
    linewidth=2,
    color = "black",
    label="Baseline value"
)

ax_xi.set_xlabel(r"$\xi$", fontsize=13)

ax_xi.set_ylabel(r"Dominant Floquet multiplier", fontsize=13)

ax_xi.set_title("(a)", fontsize=12)

ax_xi.legend(fontsize=11)

ax_xi.grid(alpha=0.25)


# ------------------------------------------------------------
# (b) eta sensitivity
# ------------------------------------------------------------

subset_eta = sensitivity_df[
    sensitivity_df["Parameter"] == "eta"
]

eta_baseline = PARAMS["eta"]

ax_eta.plot(
    subset_eta["Parameter_value"],
    subset_eta["rho_R2"],
    marker="o",
    linewidth=2,
    label=r"Floquet multiplier"
)

ax_eta.axhline(
    y=1.0,
    linestyle="--",
    linewidth=2,
    color = "orange",
    label=r"Stability boundary = 1"
)

ax_eta.axvline(
    x=eta_baseline,
    linestyle=":",
    linewidth=2,
    color = "black",
    label="Baseline value"
)

ax_eta.set_xlabel(r"$\eta$", fontsize=13)

ax_eta.set_ylabel(r"Dominant Floquet multiplier", fontsize=13)

ax_eta.set_title("(b)", fontsize=12)

ax_eta.legend(fontsize=11)

ax_eta.grid(alpha=0.25)


# ------------------------------------------------------------
# (c) stem-cell dose b
# ------------------------------------------------------------

ax_b.plot(
    dose_df["b"],
    dose_df["rho_R2"],
    marker="o",
    linewidth=2,
    label=r"Floquet multiplier"
)

ax_b.axhline(
    y=1.0,
    linestyle="--",
    linewidth=2,
    color = "orange",
    label=r"Stability boundary = 1"
)

ax_b.axvline(
    x=BASELINE_B,
    linestyle=":",
    linewidth=2,
    color = "black",
    label="Baseline dose"
)

ax_b.set_xlabel(r"Stem-cell dose $b$ (cells/mm$^3$)", fontsize=13)

ax_b.set_ylabel(r"Dominant Floquet multiplier",fontsize=13)

ax_b2 = ax_b.twinx()

ax_b2.plot(
    dose_df["b"],
    dose_df["K_post_impulse"],
    marker="s",
    linewidth=2,
    color= "brown",
    label=r"Post impulse keratinocyte density"
)

ax_b2.axhline(
    y=K_THRESHOLD,
    linestyle=":",
    linewidth=1.5,
    color = "green",
    label=r"Healthy threshold $\tilde K=200$"
)

ax_b2.set_ylabel(r"Keratinocyte density $(mm^{-3})$", fontsize=13)

ax_b.set_title("(c)", fontsize=12)

lines_b1, labels_b1 = ax_b.get_legend_handles_labels()
lines_b2, labels_b2 = ax_b2.get_legend_handles_labels()

ax_b.legend(
    lines_b1 + lines_b2,
    labels_b1 + labels_b2,
    loc="center",
    bbox_to_anchor=(0.72, 0.72),
    fontsize=11
)

ax_b.grid(
    alpha=0.25
)


# ------------------------------------------------------------
# (d) infusion interval tau
# ------------------------------------------------------------

ax_tau.plot(
    tau_df["tau"],
    tau_df["rho_R2"],
    marker="o",
    linewidth=2,
    label=r"Floquet multiplier"
)

ax_tau.axhline(
    y=1.0,
    linestyle="--",
    color = "orange",
    linewidth=2,
    label=r"Stability boundary"
)

ax_tau.axvline(
    x=BASELINE_TAU,
    linestyle=":",
    linewidth=2,
    label="Baseline interval"
)

ax_tau.set_xlabel(r"Infusion interval $\tau$ (days)", fontsize=13)

ax_tau.set_ylabel(r"Dominant Floquet multiplier", fontsize=13)

ax_tau2 = ax_tau.twinx()

ax_tau2.plot(
    tau_df["tau"],
    tau_df["K_post_impulse"],
    marker="s",
    linewidth=2,
    color = "brown",
    label=r"Post impulse keratinocyte density"
)

ax_tau2.axhline(
    y=K_THRESHOLD,
    linestyle=":",
    linewidth=2,
    color = "green",
    label=r"Healthy threshold $\tilde K=200$"
)

ax_tau2.set_ylabel(r"Keratinocyte density $(mm^{-3})$", fontsize=13)

ax_tau.set_title("(d)", fontsize=12)

lines_tau1, labels_tau1 = ax_tau.get_legend_handles_labels()
lines_tau2, labels_tau2 = ax_tau2.get_legend_handles_labels()

ax_tau.legend(lines_tau1 + lines_tau2,labels_tau1 + labels_tau2,loc="center",fontsize=11, bbox_to_anchor=(0.72, 0.35))

ax_tau.grid(alpha=0.25)


# ------------------------------------------------------------
# Final layout and save
# ------------------------------------------------------------

fig.tight_layout()

fig.savefig("four_floquet_multiplier_analysis.png",dpi=600,bbox_inches="tight")

plt.show()


# ============================================================
# 24. SAVE FOUR-PANEL FIGURE DATA
# ============================================================

sensitivity_df.to_csv("xi_eta_floquet_sensitivity.csv",index=False)

dose_df.to_csv("dose_b_floquet_analysis.csv",index=False)

tau_df.to_csv("tau_floquet_analysis.csv",index=False)


# ============================================================
# 25. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    f"Baseline b = {BASELINE_B:.2f}, "
    f"tau = {BASELINE_TAU:.2f}"
)

print(
    f"Baseline xi = {PARAMS['xi']:.6f}, "
    f"eta = {PARAMS['eta']:.6f}"
)

print(
    f"Baseline rho_R2 = "
    f"{baseline['rho_R2']:.10e}"
)

print(
    f"Baseline full spectral radius = "
    f"{baseline['rho_full']:.10e}"
)

print(
    f"Baseline K* = "
    f"{baseline['x_post'][5]:.10f}"
)

print("\nGenerated files:")
print("1. baseline_floquet_multipliers.csv")
print("2. xi_eta_floquet_sensitivity.csv")
print("3. dose_b_floquet_analysis.csv")
print("4. tau_floquet_analysis.csv")
print("5. four_panel_floquet_sensitivity_analysis.png")
