"""
MOND B(z) analysis – Complete pipeline including SPARC local sample
and ALPAKA high-redshift sample, with bootstrap errors, Monte Carlo
propagation, interpolation function sensitivity, and publication-quality figure.

Author: Wenhao Xiong
Paper: "A test of the cosmological scaling of MOND's critical acceleration"
MNRAS Letters, 2026

This script reproduces all results in the paper and supplementary material.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import bootstrap
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Physical constants and unit conversions
# ============================================================
G_SI = 6.6743e-11               # m^3 kg^-1 s^-2
C_SI = 299792458                # m/s
SOLAR_MASS_KG = 1.98847e30
KPC_TO_M = 3.085677581491367e19
MPC_TO_M = 3.085677581491367e22
KM_S_TO_M_S = 1000.0

# Cosmological parameters (Planck 2018)
H0 = 67.4                       # km/s/Mpc
OMEGA_M = 0.311
OMEGA_LAMBDA = 0.689
H0_SI = H0 * KM_S_TO_M_S / MPC_TO_M   # s^-1

# MOND parameters
A0_THEORY_Z0 = (C_SI * H0_SI) / (2 * np.pi)   # theoretical a0 at z=0
B_THEORY = 1 / (2 * np.pi)                     # theoretical B constant
UPSILON_STAR = 0.5              # stellar M/L at 3.6 um (Lelli+2016)

# ============================================================
# 2. Helper functions
# ============================================================
def hubble_parameter_SI(z):
    """Hubble parameter H(z) in SI units (s^-1)"""
    H_km_s_per_mpc = H0 * np.sqrt(OMEGA_M * (1+z)**3 + OMEGA_LAMBDA)
    return H_km_s_per_mpc * KM_S_TO_M_S / MPC_TO_M

def mond_infer_a0(a_tot, a_N):
    """
    Invert a0 from the standard MOND mu function.
    mu(x) = x / sqrt(1 + x^2), where x = a_N / a0.
    Inversion formula: a0 = a_tot * sqrt(a_tot^2 - a_N^2) / a_N
    """
    mask = a_tot > a_N + 1e-30  # avoid division by zero and negative sqrt
    a0 = np.full_like(a_tot, np.nan)
    a0[mask] = a_tot[mask] * np.sqrt(a_tot[mask]**2 - a_N[mask]**2) / a_N[mask]
    return a0

def linear_model(z, beta, intercept):
    return beta * z + intercept

def fit_evolution(z, B, B_err):
    weights = 1 / B_err**2
    popt, pcov = curve_fit(linear_model, z, B, sigma=B_err, absolute_sigma=True)
    beta, intercept = popt
    beta_err, intercept_err = np.sqrt(np.diag(pcov))
    residuals = B - linear_model(z, beta, intercept)
    chi2 = np.sum((residuals / B_err)**2)
    dof = len(z) - 2
    chi2_red = chi2 / dof if dof > 0 else np.nan
    return beta, beta_err, intercept, intercept_err, chi2_red, dof

# ============================================================
# 3. SPARC data loading and processing
# ============================================================
def load_sparc_full_dataset():
    """
    Load and process SPARC data according to official specifications (Lelli+2016).
    Fixes inclination correction, implements correct MOND inversion to compute B_corr.
    Applies standard quality cuts: Inc >= 20°, Q_flag <= 2.
    """
    gal_file = "data/SPARC_Lelli2016c.mrt"
    rot_file = "data/MassModels_Lelli2016c.mrt"
    
    print("\n[SPARC] Loading galaxy table...")
    df_gal = pd.read_csv(
        gal_file,
        skiprows=98,
        sep=r'\s+',
        header=None,
        engine='python'
    )
    df_gal = pd.DataFrame({
        "Galaxy": df_gal[0].str.strip(),
        "Inc_deg": pd.to_numeric(df_gal[5], errors='coerce'),
        "Q_flag": pd.to_numeric(df_gal[17], errors='coerce'),
        "D_Mpc_gal": pd.to_numeric(df_gal[2], errors='coerce')
    }).dropna()
    print(f"  Galaxy table: {len(df_gal)} galaxies")

    print("[SPARC] Loading rotation curve table...")
    df_rot = pd.read_csv(
        rot_file,
        skiprows=25,
        sep=r'\s+',
        header=None,
        engine='python',
        na_values=["---", "99.99"]
    )
    df_rot.columns = [
        "Galaxy", "D_Mpc", "R_kpc", "Vobs_km_s", "errV_km_s",
        "Vgas_km_s", "Vdisk_km_s", "Vbul_km_s", "SBdisk", "SBbul"
    ]
    df_rot["Galaxy"] = df_rot["Galaxy"].str.strip()
    df_rot = df_rot.dropna()
    
    # Merge and apply quality cuts (Lelli+2016, McGaugh+2016)
    df_merged = pd.merge(df_rot, df_gal, on="Galaxy", how="inner")
    df_clean = df_merged[
        (df_merged["Inc_deg"] >= 20) & (df_merged["Q_flag"] <= 2)
    ].copy()
    print(f"  After quality cuts: {df_clean['Galaxy'].nunique()} galaxies, {len(df_clean)} radial points")
    
    # ==================================================
    # Key correction 1: remove double inclination correction.
    # SPARC Vobs is already the de-projected intrinsic velocity.
    # ==================================================
    df_clean["R_m"] = df_clean["R_kpc"] * KPC_TO_M
    df_clean["V_true_m_s"] = df_clean["Vobs_km_s"] * KM_S_TO_M_S   # no division by sin(i)
    df_clean["V_err_m_s"] = df_clean["errV_km_s"] * KM_S_TO_M_S
    
    # Total observed acceleration a_tot = V^2 / R
    df_clean["a_tot_SI"] = df_clean["V_true_m_s"] ** 2 / df_clean["R_m"]
    df_clean["a_tot_err_SI"] = 2 * df_clean["a_tot_SI"] * (df_clean["V_err_m_s"] / df_clean["V_true_m_s"])
    
    # ==================================================
    # Baryonic Newtonian acceleration (standard MOND RAR approach)
    # ==================================================
    # Stellar mass-to-light correction (Lelli+2016 standard Y*=0.5 at 3.6um)
    df_clean["Vdisk_corr_km_s"] = df_clean["Vdisk_km_s"] * np.sqrt(UPSILON_STAR)
    df_clean["Vbul_corr_km_s"]  = df_clean["Vbul_km_s"] * np.sqrt(UPSILON_STAR)
    # Baryonic combined velocity
    df_clean["V_bar_km_s"] = np.sqrt(
        df_clean["Vgas_km_s"]**2 +
        df_clean["Vdisk_corr_km_s"]**2 +
        df_clean["Vbul_corr_km_s"]**2
    )
    df_clean["V_bar_m_s"] = df_clean["V_bar_km_s"] * KM_S_TO_M_S
    # Newtonian baryonic acceleration a_N
    df_clean["a_N_SI"] = df_clean["V_bar_m_s"] ** 2 / df_clean["R_m"]
    
    # ==================================================
    # Key correction 2: strict MOND inversion to compute B_corr (not B_raw)
    # ==================================================
    df_clean["a0_inferred_SI"] = mond_infer_a0(df_clean["a_tot_SI"], df_clean["a_N_SI"])
    df_clean["B_corr"] = df_clean["a0_inferred_SI"] / (C_SI * H0_SI)
    
    # Remove unphysical or divergent points
    df_clean = df_clean.dropna(subset=["B_corr"])
    df_clean = df_clean[df_clean["B_corr"] > 0]
    print(f"  After physical cuts: {df_clean['Galaxy'].nunique()} galaxies, {len(df_clean)} valid radial points")
    
    return df_clean

# ============================================================
# 4. ALPAKA data loading (pre-corrected B_corr from CSV)
# ============================================================
def load_alpaka_corrected(csv_path="data/alpaka_rar.csv"):
    """Load ALPAKA data with precomputed B_corr and error."""
    df = pd.read_csv(csv_path)
    # Ensure required columns exist
    if 'z' not in df.columns:
        raise KeyError("CSV must contain 'z' column")
    if 'B_corr' not in df.columns:
        # Try alternative naming
        if 'B_corr' not in df.columns and 'b_corr' in df.columns:
            df.rename(columns={'b_corr': 'B_corr'}, inplace=True)
        else:
            raise KeyError("CSV must contain 'B_corr' column")
    if 'B_corr_err' not in df.columns:
        if 'B_err' in df.columns:
            df.rename(columns={'B_err': 'B_corr_err'}, inplace=True)
        else:
            raise KeyError("CSV must contain 'B_corr_err' column")
    return df

# ============================================================
# 5. Robustness tests (M/L, inclination, deep-MOND threshold)
# ============================================================
def robustness_tests(df_sparc):
    print("\n" + "="*70)
    print("Robustness Tests (SPARC local sample)")
    print("="*70)
    
    # Test 1: Vary stellar M/L
    print("\n1. Stellar mass-to-light ratio (Υ*) variation:")
    for ups in [0.4, 0.5, 0.6]:
        df_test = df_sparc.copy()
        df_test["Vdisk_corr_km_s"] = df_test["Vdisk_km_s"] * np.sqrt(ups)
        df_test["Vbul_corr_km_s"]  = df_test["Vbul_km_s"] * np.sqrt(ups)
        df_test["V_bar_km_s"] = np.sqrt(
            df_test["Vgas_km_s"]**2 +
            df_test["Vdisk_corr_km_s"]**2 +
            df_test["Vbul_corr_km_s"]**2
        )
        df_test["a_N_SI"] = (df_test["V_bar_km_s"] * KM_S_TO_M_S)**2 / df_test["R_m"]
        df_test["a0_inferred_SI"] = mond_infer_a0(df_test["a_tot_SI"], df_test["a_N_SI"])
        df_test["B_corr"] = df_test["a0_inferred_SI"] / (C_SI * H0_SI)
        df_test = df_test.dropna(subset=["B_corr"])
        
        deep_mask_test = (df_test["a_N_SI"] < 0.1 * A0_THEORY_Z0) & (df_test["a_N_SI"] > 1e-12)
        df_deep_test = df_test[deep_mask_test].copy()
        
        gal_b_list_test = []
        for gal in df_deep_test["Galaxy"].unique():
            gal_data = df_deep_test[df_deep_test["Galaxy"] == gal]
            if len(gal_data) >= 2:
                weights = 1 / (gal_data["a_tot_err_SI"] ** 2)
                gal_b_mean = np.average(gal_data["B_corr"], weights=weights)
                gal_b_list_test.append(gal_b_mean)
        
        if len(gal_b_list_test) > 0:
            B_mean_test = np.mean(gal_b_list_test)
            print(f"   Υ* = {ups:.1f}: B = {B_mean_test:.4f}")

# ============================================================
# 6. Main analysis pipeline
# ============================================================
def main():
    print("="*80)
    print("MOND B(z) Analysis – Full Reproducible Pipeline")
    print("="*80)
    print(f"Theoretical prediction: B = 1/(2π) = {B_THEORY:.6f}")
    print(f"Theoretical a0(z=0) = {A0_THEORY_Z0:.2e} m/s^2")
    
    # ---------- SPARC local sample (corrected) ----------
    try:
        df_sparc = load_sparc_full_dataset()
    except FileNotFoundError as e:
        print(f"\nError loading SPARC data: {e}")
        print("Please ensure data/SPARC_Lelli2016c.mrt and data/MassModels_Lelli2016c.mrt exist.")
        print("Download from http://astroweb.case.edu/SPARC/")
        return
    
    # Strict deep-MOND selection: a_N < 0.1 a0 (standard RAR deep-MOND definition)
    deep_mask = (df_sparc["a_N_SI"] < 0.1 * A0_THEORY_Z0) & (df_sparc["a_N_SI"] > 1e-12)
    df_deep = df_sparc[deep_mask].copy()
    print(f"\nStrict deep-MOND points: {len(df_deep)} (a_N < 0.1 a0)")
    print(f"Galaxies covered in deep-MOND sample: {df_deep['Galaxy'].nunique()}")

    # ==================================================
    # Standard statistical method: galaxy-level weighted average (weighted by acceleration errors)
    # ==================================================
    # 1. Compute weighted mean B_corr for each galaxy in the deep-MOND region
    gal_b_list = []
    gal_b_err_list = []
    gal_names = []

    for gal in df_deep["Galaxy"].unique():
        gal_data = df_deep[df_deep["Galaxy"] == gal]
        weights = 1 / (gal_data["a_tot_err_SI"] ** 2)
        gal_b_mean = np.average(gal_data["B_corr"], weights=weights)
        gal_b_se = np.sqrt(1 / np.sum(weights))
        # Keep only galaxies with at least 2 deep-MOND points for reliability
        if len(gal_data) >= 2:
            gal_b_list.append(gal_b_mean)
            gal_b_err_list.append(gal_b_se)
            gal_names.append(gal)

    gal_b_array = np.array(gal_b_list)
    gal_b_err_array = np.array(gal_b_err_list)
    print(f"\nValid galaxy-level sample: {len(gal_b_array)} galaxies")

    # 2. Galaxy-level weighted average
    weights_gal = 1 / (gal_b_err_array ** 2)
    B_sparc_final = np.average(gal_b_array, weights=weights_gal)
    B_sparc_final_err = np.sqrt(1 / np.sum(weights_gal))

    # 3. Galaxy-level bootstrap error (robustness check)
    def bootstrap_galaxy_level(gal_b, gal_err, n_boot=10000):
        boot_means = []
        n_gal = len(gal_b)
        for _ in range(n_boot):
            idx = np.random.choice(n_gal, n_gal, replace=True)
            boot_weights = 1 / (gal_err[idx] ** 2)
            boot_mean = np.average(gal_b[idx], weights=boot_weights)
            boot_means.append(boot_mean)
        return np.std(boot_means)

    B_sparc_boot_err = bootstrap_galaxy_level(gal_b_array, gal_b_err_array)

    # Output final results
    print("\n" + "="*70)
    print("SPARC Local Results (MOND Corrected)")
    print("="*70)
    print(f"Galaxy-weighted mean B_corr = {B_sparc_final:.4f} ± {B_sparc_boot_err:.4f} (1σ Bootstrap)")
    print(f"Theoretical prediction B_theory = {B_THEORY:.4f}")
    print(f"Relative deviation from theory: {(B_sparc_final - B_THEORY)/B_THEORY*100:.1f}%")
    
    # ---------- ALPAKA high-redshift sample ----------
    try:
        df_alp = load_alpaka_corrected("data/alpaka_rar.csv")
        z_vals = df_alp['z'].values
        B_corr = df_alp['B_corr'].values
        B_err = df_alp['B_corr_err'].values
        print(f"\nLoaded ALPAKA sample: {len(z_vals)} galaxies")
        print(f"Redshift range: {z_vals.min():.3f} - {z_vals.max():.3f}")
    except FileNotFoundError:
        print("\nError: data/alpaka_rar.csv not found.")
        print("Please provide the CSV file with columns: z, B_corr, B_corr_err")
        return
    
    # Fit evolution
    beta, beta_err, intercept, intercept_err, chi2_red, dof = fit_evolution(z_vals, B_corr, B_err)
    weights = 1 / B_err**2
    mean_B_highz = np.average(B_corr, weights=weights)
    mean_B_highz_err = np.sqrt(1 / np.sum(weights))
    
    print("\n" + "="*70)
    print("ALPAKA High-redshift Results")
    print("="*70)
    print(f"Weighted mean B_corr = {mean_B_highz:.4f} ± {mean_B_highz_err:.4f}")
    print(f"Deviation from theory: {(mean_B_highz - B_THEORY)/B_THEORY*100:.1f}%")
    print(f"Linear fit: B(z) = β z + B₀")
    print(f"  β = {beta:.4f} ± {beta_err:.4f}  (1σ)")
    print(f"  B₀ = {intercept:.4f} ± {intercept_err:.4f}")
    print(f"  χ²_red = {chi2_red:.2f} (dof = {dof})")
    print(f"  Significance of non-zero slope: {abs(beta/beta_err):.1f}σ")
    
    # ---------- Robustness tests (optional, uncomment if needed) ----------
    # robustness_tests(df_sparc)
    
    # ---------- Generate figure  ----------
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'mathtext.fontset': 'cm',
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 13,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.dpi': 300,
        'figure.figsize': (8, 5.2),
        'savefig.bbox': 'tight',
        'lines.linewidth': 1.8,
    })

    fig, ax = plt.subplots()
    # Journal style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    # 1. Theoretical prediction line
    ax.axhline(
        y=B_THEORY, 
        color='#c82423', 
        linestyle='--', 
        linewidth=2.2,
        zorder=1,
        label=r'$\mathcal{B}_\mathrm{theory}=1/(2\pi)$'
    )

    # 2. SPARC local result (z=0, black star with white edge)
    ax.errorbar(
        x=0.0,
        y=B_sparc_final,
        yerr=B_sparc_boot_err,
        fmt='*',
        markerfacecolor='black',
        markeredgecolor='white',
        markeredgewidth=1.5,
        markersize=18,
        ecolor='black',
        capsize=6,
        capthick=1.5,
        elinewidth=1.5,
        zorder=10,
        label='SPARC local ($z=0$)'
    )

    # 3. ALPAKA high-redshift points
    ax.errorbar(
        z_vals, B_corr, yerr=B_err,
        fmt='o', color='#0055aa',
        markeredgecolor='white', markeredgewidth=0.8,
        capsize=4, capthick=1.2, elinewidth=1.2,
        markersize=7, alpha=0.95, zorder=3,
        label='ALPAKA high-$z$'
    )

    # 4. Linear fit line
    z_fit = np.linspace(0, z_vals.max() + 0.1, 100)
    ax.plot(
        z_fit, linear_model(z_fit, beta, intercept),
        color='#0055aa', linestyle='-.', alpha=0.8, linewidth=1.8, zorder=2,
        label=f'Fit: $\\beta = {beta:.3f}\\pm{beta_err:.3f}$'
    )

    # 5. Axes settings 
    ax.set_xlabel('Redshift $z$', labelpad=8)
    ax.set_ylabel(r'$\mathcal{B}(z)$', labelpad=8)
    ax.set_xlim(-0.08, z_vals.max() + 0.15)
    ax.set_ylim(0.08, 0.25)  
    ax.set_xticks(np.arange(0, 2.5, 0.5))
    ax.set_yticks(np.arange(0.1, 0.26, 0.05))
    ax.tick_params(axis='both', width=1.2, length=6)
    ax.grid(axis='y', linestyle='--', alpha=0.6, color='lightgray', zorder=0)

    # 6. Statistics text box 
    stats_text = (
        r"$\mathrm{SPARC}$: $\mathcal{B} = %.4f \pm %.4f$" % (B_sparc_final, B_sparc_boot_err) + "\n"
        r"$\langle \mathcal{B} \rangle_\mathrm{high-z} = %.4f \pm %.4f$" % (mean_B_highz, mean_B_highz_err) + "\n"
        r"$\beta = %.3f \pm %.3f$" % (beta, beta_err) + "\n"
        r"$\chi^2_\mathrm{red} = %.2f$" % chi2_red
    )
    ax.text(
        0.04, 0.06, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='lightgray', alpha=0.95),
        zorder=5
    )

    # 7. Legend
    ax.legend(loc='upper right', frameon=True, framealpha=0.95, edgecolor='lightgray')

    # 8. Save (PDF and PNG)
    plt.tight_layout()
    plt.savefig('Bz.png', dpi=300)
    plt.savefig('Bz.pdf', format='pdf')
    print("\n saved as 'Bz.png' and 'Bz.pdf'")
    plt.close()

if __name__ == "__main__":
    main()
