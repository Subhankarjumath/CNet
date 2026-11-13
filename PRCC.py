# ============================================================
# GLOBAL SENSITIVITY ANALYSIS OF R_E
# Based on Appendix (1) of the manuscript
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from scipy.stats import rankdata, t
from scipy.stats import qmc


# ============================================================
# 1. BASELINE PARAMETERS
# ============================================================

BASE = {
    "rho0": 0.10,

    "b1": 0.55,
    "b2": 0.35,

    "l1": 0.47,
    "l2": 0.60,

    "lambda1": 0.125,
    "lambda2": 0.343,
    "lambdaM": 1.10,

    "CT": 15.0,
    "CD": 20.0,
    "CK": 30.0,

    "gamma": 0.001,
    "delta1": 0.0002,
    "delta2": 0.001,

    "xi": 0.7,
    "eta": 2.90,

    "lambdaT": 0.07,
    "lambdaD": 0.15,
    "lambdaK": 0.20,
}


# ============================================================
# 2. PARAMETERS INCLUDED IN THE PRCC ANALYSIS
# ============================================================

PARAMETER_NAMES = [
    "rho0",
    "b1",
    "b2",
    "l1",
    "l2",
    "lambda1",
    "lambda2",
    "lambdaM",
    "CT",
    "CD",
    "CK",
    "gamma",
    "delta1",
    "delta2",
    "xi",
    "eta",
    "lambdaT",
    "lambdaD",
    "lambdaK",
]

# ============================================================
# 14A. PARAMETER SYMBOLS FOR MANUSCRIPT FIGURE
# ============================================================

parameter_symbols = {
    "rho0": r"$\rho_0$",

    "b1": r"$b_1$",
    "b2": r"$b_2$",

    "l1": r"$l_1$",
    "l2": r"$l_2$",

    "lambda1": r"$\lambda_1$",
    "lambda2": r"$\lambda_2$",
    "lambdaM": r"$\lambda_M$",

    "CT": r"$C_T$",
    "CD": r"$C_D$",
    "CK": r"$C_K$",

    "gamma": r"$\gamma$",
    "delta1": r"$\delta_1$",
    "delta2": r"$\delta_2$",

    "xi": r"$\xi$",
    "eta": r"$\eta$",

    "lambdaT": r"$\lambda_T$",
    "lambdaD": r"$\lambda_D$",
    "lambdaK": r"$\lambda_K$"
}
# ============================================================
# 3. PARAMETER RANGES
#
# Here we use ±30% around the baseline.
# Change LOWER_FACTOR / UPPER_FACTOR if required.
# ============================================================

LOWER_FACTOR = 0.70
UPPER_FACTOR = 1.30


def parameter_bounds():

    bounds = {}

    for name in PARAMETER_NAMES:

        low = BASE[name] * LOWER_FACTOR
        high = BASE[name] * UPPER_FACTOR

        bounds[name] = (low, high)

    # Biological constraint:
    # 0 < b1,b2 < 1
    bounds["b1"] = (
        max(0.05, BASE["b1"] * LOWER_FACTOR),
        min(0.95, BASE["b1"] * UPPER_FACTOR)
    )

    bounds["b2"] = (
        max(0.05, BASE["b2"] * LOWER_FACTOR),
        min(0.95, BASE["b2"] * UPPER_FACTOR)
    )

    return bounds


BOUNDS = parameter_bounds()


# ============================================================
# 4. DERIVED PARAMETERS
#
# sigma_i = lambda_i - (2 b_i - 1) l_i
# rho_i   = 2(1-b_i)l_i
# ============================================================

def derived_parameters(p):

    p = p.copy()

    p["sigma1"] = (
        p["lambda1"]
        - (2.0 * p["b1"] - 1.0) * p["l1"]
    )

    p["sigma2"] = (
        p["lambda2"]
        - (2.0 * p["b2"] - 1.0) * p["l2"]
    )

    p["rho1"] = (
        2.0 * (1.0 - p["b1"]) * p["l1"]
    )

    p["rho2"] = (
        2.0 * (1.0 - p["b2"]) * p["l2"]
    )

    return p


# ============================================================
# 5. CHECK BASIC PARAMETER VALIDITY
# ============================================================

def valid_parameters(p):

    p = derived_parameters(p)

    if p["sigma1"] <= 0:
        return False

    if p["sigma2"] <= 0:
        return False

    if p["lambdaM"] <= 0:
        return False

    return True


# ============================================================
# 6. SYSTEM (8) EQUILIBRIUM
#
# First three equilibrium coordinates have explicit forms:
#
# S1* = rho0/sigma1
# S2* = rho1*S1*/sigma2
# SM* = rho2*S2*/lambdaM
#
# The remaining three variables satisfy:
#
# 0 = CT - gamma*T*D - delta1*T*K
#       - xi*T*SM - lambdaT*T
#
# 0 = CD - delta2*D*K - lambdaD*D
#
# 0 = CK + (delta1*T*K + delta2*D*K)/(1+eta*SM)
#       - lambdaK*K
# ============================================================

def equilibrium(p):

    p = derived_parameters(p)

    # -----------------------------------------
    # Stem-cell equilibrium
    # -----------------------------------------

    S1 = p["rho0"] / p["sigma1"]

    S2 = (
        p["rho1"] * S1
        / p["sigma2"]
    )

    SM = (
        p["rho2"] * S2
        / p["lambdaM"]
    )

    if min(S1, S2, SM) <= 0:
        return None

    # -----------------------------------------
    # Equations for T, D, K
    # -----------------------------------------

    def residual(log_state):

        T, D, K = np.exp(log_state)

        eq1 = (
            p["CT"]
            - p["gamma"] * T * D
            - p["delta1"] * T * K
            - p["xi"] * T * SM
            - p["lambdaT"] * T
        )

        eq2 = (
            p["CD"]
            - p["delta2"] * D * K
            - p["lambdaD"] * D
        )

        eq3 = (
            p["CK"]
            + (
                p["delta1"] * T * K
                + p["delta2"] * D * K
            ) / (1.0 + p["eta"] * SM)
            - p["lambdaK"] * K
        )

        # scale residuals
        return np.array([
            eq1 / max(p["CT"], 1e-12),
            eq2 / max(p["CD"], 1e-12),
            eq3 / max(p["CK"], 1e-12),
        ])

    # -----------------------------------------
    # Initial guess
    # -----------------------------------------

    x0 = np.log([
        max(p["CT"] / p["lambdaT"], 1e-3),
        max(p["CD"] / p["lambdaD"], 1e-3),
        max(p["CK"] / p["lambdaK"], 1e-3),
    ])

    # -----------------------------------------
    # Positive least-squares solution
    # -----------------------------------------

    result = least_squares(
        residual,
        x0,
        max_nfev=3000,
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11
    )

    if not result.success:
        return None

    T, D, K = np.exp(result.x)

    # Check residual accuracy
    if np.max(np.abs(residual(result.x))) > 1e-6:
        return None

    if min(T, D, K) <= 0:
        return None

    return np.array([
        S1,
        S2,
        SM,
        T,
        D,
        K
    ])


# ============================================================
# 7. INVARIANT-REGION BOUNDS G1,...,G6
#
# According to Appendix (1):
#
# G1 = rho0/sigma1
# G2 = rho0*rho1/(sigma1*sigma2)
# G3 = rho0*rho1*rho2/(sigma1*sigma2*lambdaM)
# G4 = CT/lambdaT
# G5 = CD/lambdaD
#
# For G6 we use the conservative numerical choice
# SM_min = 0, as explained below.
# ============================================================

def invariant_bounds(p):

    p = derived_parameters(p)

    G1 = p["rho0"] / p["sigma1"]

    G2 = (
        p["rho0"]
        * p["rho1"]
        / (p["sigma1"] * p["sigma2"])
    )

    G3 = (
        p["rho0"]
        * p["rho1"]
        * p["rho2"]
        / (
            p["sigma1"]
            * p["sigma2"]
            * p["lambdaM"]
        )
    )

    G4 = p["CT"] / p["lambdaT"]

    G5 = p["CD"] / p["lambdaD"]

    # Conservative lower bound:
    SM_min = 0.0

    denominator = (
        p["lambdaK"] * (1.0 + p["eta"] * SM_min)
        - (
            p["delta1"] * G4
            + p["delta2"] * G5
        )
    )

    if denominator <= 0:
        return None

    G6 = (
        p["CK"] * (1.0 + p["eta"] * SM_min)
        / denominator
    )

    return {
        "G1": G1,
        "G2": G2,
        "G3": G3,
        "G4": G4,
        "G5": G5,
        "G6": G6,
    }


# ============================================================
# 8. CALCULATE R_E
#
# We operationalize the barred variables using the
# invariant-region upper bounds:
#
# Sbar1 = G1
# Sbar2 = G2
# SbarM = G3
# Tbar  = G4
# Dbar  = G5
# Kbar  = G6
#
# Lower bounds:
# S_M_min = T_min = D_min = 0
#
# This is explicitly a conservative numerical
# implementation of the Appendix criterion.
# ============================================================

def calculate_RE(p, equilibrium_state):

    p = derived_parameters(p)

    eq = invariant_bounds(p)

    if eq is None:
        return np.nan

    S1_star, S2_star, SM_star, T_star, D_star, K_star = (
        equilibrium_state
    )

    G1 = eq["G1"]
    G2 = eq["G2"]
    G3 = eq["G3"]
    G4 = eq["G4"]
    G5 = eq["G5"]
    G6 = eq["G6"]

    # --------------------------------------------------------
    # Conservative lower bounds
    # --------------------------------------------------------

    SM_min = 0.0
    T_min = 0.0
    D_min = 0.0
    K_min = 0.0

    # --------------------------------------------------------
    # W terms
    # --------------------------------------------------------

    W1 = G1 - S1_star
    W2 = G2 - S2_star
    W3 = G3 - SM_star
    W4 = G6 - K_star

    # --------------------------------------------------------
    # Omega terms
    # --------------------------------------------------------

    Omega1 = (
        T_star * G5
        + T_star * G5
    )

    Omega2 = (
        T_star * G6
        + T_star * G6
    )

    Omega3 = (
        T_star * G3
        + T_star * G3
    )

    Omega4 = (
        T_star * G5
        + T_star * G5
    )

    # --------------------------------------------------------
    # Theta terms
    # --------------------------------------------------------

    Theta1 = (
        T_min * D_min
        + T_star * D_star
    )

    Theta2 = (
        T_min * K_min
        + T_star * K_star
    )

    Theta3 = (
        T_min * SM_min
        + T_star * SM_star
    )

    Theta4 = (
        D_min * K_min
        + D_star * K_star
    )

    # --------------------------------------------------------
    # Delta
    # --------------------------------------------------------

    Delta = (
        (
            p["delta1"] * G4
            + p["delta2"] * G5
        )
        / (1.0 + p["eta"] * SM_min)
        -
        (
            p["delta1"] * T_star
            + p["delta2"] * D_star
        )
        / (1.0 + p["eta"] * SM_star)
    )

    # --------------------------------------------------------
    # Numerator
    # --------------------------------------------------------

    numerator = (

        W2 * (
            p["rho1"] * W1
            + p["rho2"] * W3
        )

        + G6 * W4 * Delta

        + T_star * (
            p["gamma"] * Omega1
            + p["delta1"] * Omega2
            + p["xi"] * Omega3
        )

        + p["delta2"] * D_star * Omega4
    )

    # --------------------------------------------------------
    # Denominator
    # --------------------------------------------------------

    denominator = (

        T_star * (
            p["gamma"] * Theta1
            + p["delta1"] * Theta2
            + p["xi"] * Theta3
        )

        + p["delta2"] * D_star * Theta4
    )

    if denominator <= 0:
        return np.nan

    RE = numerator / denominator

    return RE


# ============================================================
# 9. LATIN HYPERCUBE SAMPLING
# ============================================================

def generate_LHS_samples(N=3000, seed=12345):

    sampler = qmc.LatinHypercube(
        d=len(PARAMETER_NAMES),
        seed=seed
    )

    unit_samples = sampler.random(N)

    samples = np.zeros_like(unit_samples)

    for j, name in enumerate(PARAMETER_NAMES):

        low, high = BOUNDS[name]

        samples[:, j] = (
            low
            + (high - low) * unit_samples[:, j]
        )

    return pd.DataFrame(
        samples,
        columns=PARAMETER_NAMES
    )


# ============================================================
# 10. RUN R_E SENSITIVITY ANALYSIS
# ============================================================

def run_RE_sensitivity(
    N=3000,
    seed=12345
):

    samples = generate_LHS_samples(
        N=N,
        seed=seed
    )

    results = []

    total = len(samples)

    for i, row in samples.iterrows():

        p = row.to_dict()

        if not valid_parameters(p):
            continue

        eq = equilibrium(p)

        if eq is None:
            continue

        RE = calculate_RE(
            p,
            eq
        )

        if not np.isfinite(RE):
            continue

        record = p.copy()

        record["S1_star"] = eq[0]
        record["S2_star"] = eq[1]
        record["SM_star"] = eq[2]

        record["T_star"] = eq[3]
        record["D_star"] = eq[4]
        record["K_star"] = eq[5]

        record["RE"] = RE

        results.append(record)

        if (i + 1) % 250 == 0:
            print(
                f"Processed {i+1}/{total} samples"
            )

    results_df = pd.DataFrame(results)

    print("\n------------------------------------------")
    print("R_E sensitivity analysis completed")
    print("------------------------------------------")

    print(
        f"Valid samples: {len(results_df)} / {N}"
    )

    if len(results_df) > 0:

        print(
            f"R_E minimum = "
            f"{results_df['RE'].min():.6f}"
        )

        print(
            f"R_E median  = "
            f"{results_df['RE'].median():.6f}"
        )

        print(
            f"R_E maximum = "
            f"{results_df['RE'].max():.6f}"
        )

        stable_fraction = (
            np.mean(results_df["RE"] < 1.0)
        )

        print(
            f"Fraction with R_E < 1 = "
            f"{stable_fraction:.3f}"
        )

    return results_df


# ============================================================
# 11. PRCC CALCULATION
# ============================================================

def residualize(target, controls):

    """
    Residualize target with respect to controls
    using ordinary least squares.
    """

    if controls.shape[1] == 0:
        return target.copy()

    X = np.column_stack([
        np.ones(len(target)),
        controls
    ])

    beta = np.linalg.lstsq(
        X,
        target,
        rcond=None
    )[0]

    fitted = X @ beta

    return target - fitted


def partial_corr(x, y, controls):

    """
    Partial correlation between x and y
    controlling for all variables in controls.
    """

    rx = residualize(x, controls)
    ry = residualize(y, controls)

    denominator = (
        np.sqrt(np.sum(rx**2))
        * np.sqrt(np.sum(ry**2))
    )

    if denominator == 0:
        return np.nan

    return (
        np.sum(rx * ry)
        / denominator
    )


def calculate_PRCC(df):

    X = df[PARAMETER_NAMES].values
    Y = df["RE"].values

    # Rank transformation
    X_rank = np.column_stack([
        rankdata(X[:, j])
        for j in range(X.shape[1])
    ])

    Y_rank = rankdata(Y)

    output = []

    n = len(df)
    p = len(PARAMETER_NAMES)

    for j, parameter in enumerate(PARAMETER_NAMES):

        control_indices = [
            k for k in range(p)
            if k != j
        ]

        X_j = X_rank[:, j]

        controls = X_rank[:, control_indices]

        prcc = partial_corr(
            X_j,
            Y_rank,
            controls
        )

        # t statistic
        if np.isfinite(prcc) and abs(prcc) < 1:

            t_stat = (
                prcc
                * np.sqrt(
                    (n - p - 1)
                    / (1.0 - prcc**2)
                )
            )

            p_value = (
                2.0
                * t.sf(
                    abs(t_stat),
                    df=n - p - 1
                )
            )

        else:

            t_stat = np.nan
            p_value = np.nan

        output.append({
            "Parameter": parameter,
            "PRCC": prcc,
            "p_value": p_value,
            "abs_PRCC": abs(prcc)
        })

    prcc_df = pd.DataFrame(output)

    prcc_df = (
        prcc_df
        .sort_values(
            "abs_PRCC",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return prcc_df


# ============================================================
# 12. MAIN ANALYSIS
# ============================================================

results_df = run_RE_sensitivity(
    N=3000,
    seed=2026
)


# ============================================================
# 13. CALCULATE PRCC
# ============================================================

if len(results_df) > len(PARAMETER_NAMES) + 5:

    prcc_df = calculate_PRCC(
        results_df
    )

else:

    raise RuntimeError(
        "Too few valid samples for PRCC analysis."
    )


# ============================================================
# 14. PRINT PRCC TABLE
# ============================================================

print("\n==========================================")
print("PRCC ranking for R_E")
print("==========================================")

print(
    prcc_df[
        [
            "Parameter",
            "PRCC",
            "p_value"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 15. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "RE_LHS_samples.csv",
    index=False
)

prcc_df.to_csv(
    "RE_PRCC_results.csv",
    index=False
)


# ============================================================
# 16. PRCC BAR PLOT
# ============================================================

plot_df = prcc_df.copy()

# Reverse order so largest |PRCC| is at top
plot_df = plot_df.iloc[::-1]

# Convert parameter names to mathematical symbols
plot_df["Parameter_symbol"] = (
    plot_df["Parameter"].map(parameter_symbols)
)

fig, ax = plt.subplots(
    figsize=(9, 5)
)

ax.barh(
    plot_df["Parameter_symbol"],
    plot_df["PRCC"]
)

ax.axvline(
    0,
    linestyle="--",
    color="black",
    linewidth=2
)

ax.set_xlabel(
    "Partial Rank Correlation Coefficient (PRCC)",
    fontsize=12
)

ax.set_ylabel(
    "Model parameter",
    fontsize=12
)

ax.tick_params(
    axis="both",
    labelsize=12
)

ax.grid(
    axis="x",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    "RE_PRCC_global_sensitivity.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 17. SIGNIFICANT PARAMETERS ONLY
# ============================================================

significant_df = prcc_df[
    prcc_df["p_value"] < 0.5
].copy()

print("\n==========================================")
print("Statistically significant parameters")
print("==========================================")

print(
    significant_df[
        [
            "Parameter",
            "PRCC",
            "p_value"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)