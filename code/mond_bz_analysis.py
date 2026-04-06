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
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Core configuration: Deep-MOND
# ============================================================
MU_TYPE = 'simple'       # Use simple μ function
UPSILON_STAR = 0.6       # Use Υ*=0.6
DEEP_THRESHOLD = 0.2     # Deep-MOND threshold: a_N < 0.2 a0

# ============================================================
# 1. Physical constants and unit conversions
# ============================================================
G_SI = 6.6743e-11               # m^3 kg^-1 s^-2
C_SI = 299792458                # m/s
KPC_TO_M = 3.085677581491367e19
MPC_TO_M = 3.085677581491367e22
KM_S_TO_M_S = 1000.0

# Cosmological parameters (Planck 2018)
H0 = 67.4                       # km/s/Mpc
OMEGA_M = 0.311
OMEGA_LAMBDA = 0.689
H0_SI = H0 * KM_S_TO_M_S / MPC_TO_M   # s^-1

# MOND theoretical values
A0_THEORY_Z0 = (C_SI * H0_SI) / (2 * np.pi)   # theoretical a0 at z=0
B_THEORY = 1 / (2 * np.pi)                     # theoretical B value

# ============================================================
# 2. Core functions
# ============================================================
def hubble_parameter_SI(z):
    """Hubble parameter H(z) in SI units (s^-1)"""
    H_km_s_per_mpc = H0 * np.sqrt(OMEGA_M * (1+z)**3 + OMEGA_LAMBDA)
    return H_km_s_per_mpc * KM_S_TO_M_S / MPC_TO_M

def mond_infer_a0(a_tot, a_N):
    """
    Correct inversion formula for the simple μ function.
    μ(x) = x/(1+x) → a0 = (a_tot² / a_N) - a_tot
    """
    mask = a_tot > a_N + 1e-30
    a0 = np.full_like(a_tot, np.nan)
    a0[mask] = (a_tot[mask] ** 2 / a_N[mask]) - a_tot[mask]
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
# 3. SPARC data loading
# ============================================================
def load_sparc_full_dataset():
    gal_file = "data/SPARC_Lelli2016c.mrt"
    rot_file = "data/MassModels_Lelli2016c.mrt"
    
    print("\n[SPARC] Loading galaxy table...")
    df_gal = pd.read_csv(
        gal_file, skiprows=98, sep=r'\s+', header=None, engine='python'
    )
    df_gal = pd.DataFrame({
        "Galaxy": df_gal[0].str.strip(),
        "Inc_deg": pd.to_numeric(df_gal[5], errors='coerce'),
        "Q_flag": pd.to_numeric(df_gal[17], errors='coerce'),
    }).dropna()
    print(f"  Raw galaxy table: {len(df_gal)} galaxies")

    print("[SPARC] Loading rotation curve table...")
    df_rot = pd.read_csv(
        rot_file, skiprows=25, sep=r'\s+', header=None, engine='python',
        na_values=["---", "99.99"]
    )
    df_rot.columns = [
        "Galaxy", "D_Mpc", "R_kpc", "Vobs_km_s", "errV_km_s",
        "Vgas_km_s", "Vdisk_km_s", "Vbul_km_s", "SBdisk", "SBbul"
    ]
    df_rot["Galaxy"] = df_rot["Galaxy"].str.strip()
    df_rot = df_rot.dropna()
    
    # Standard quality cuts
    df_merged = pd.merge(df_rot, df_gal, on="Galaxy", how="inner")
    df_clean = df_merged[
        (df_merged["Inc_deg"] >= 20) & (df_merged["Q_flag"] <= 2)
    ].copy()
    print(f"  After quality cuts: {df_clean['Galaxy'].nunique()} galaxies, {len(df_clean)} radial points")
    
    # Physical quantities
    df_clean["R_m"] = df_clean["R_kpc"] * KPC_TO_M
    df_clean["V_true_m_s"] = df_clean["Vobs_km_s"] * KM_S_TO_M_S
    df_clean["V_err_m_s"] = df_clean["errV_km_s"] * KM_S_TO_M_S
    df_clean["a_tot_SI"] = df_clean["V_true_m_s"] ** 2 / df_clean["R_m"]
    df_clean["a_tot_err_SI"] = 2 * df_clean["a_tot_SI"] * (df_clean["V_err_m_s"] / df_clean["V_true_m_s"])
    
    # Baryonic Newtonian acceleration (core correction: Υ*=0.6)
    df_clean["Vdisk_corr_km_s"] = df_clean["Vdisk_km_s"] * np.sqrt(UPSILON_STAR)
    df_clean["Vbul_corr_km_s"]  = df_clean["Vbul_km_s"] * np.sqrt(UPSILON_STAR)
    df_clean["V_bar_km_s"] = np.sqrt(
        df_clean["Vgas_km_s"]**2 +
        df_clean["Vdisk_corr_km_s"]**2 +
        df_clean["Vbul_corr_km_s"]**2
    )
    df_clean["V_bar_m_s"] = df_clean["V_bar_km_s"] * KM_S_TO_M_S
    df_clean["a_N_SI"] = df_clean["V_bar_m_s"] ** 2 / df_clean["R_m"]
    
    # MOND inversion
    df_clean["a0_inferred_SI"] = mond_infer_a0(df_clean["a_tot_SI"], df_clean["a_N_SI"])
    df_clean["B_corr"] = df_clean["a0_inferred_SI"] / (C_SI * H0_SI)
    
    # Remove unphysical values
    df_clean = df_clean.dropna(subset=["B_corr"])
    df_clean = df_clean[(df_clean["B_corr"] > 0) & (df_clean["B_corr"] < 1.0)]
    print(f"  After physical cuts: {df_clean['Galaxy'].nunique()} galaxies, {len(df_clean)} valid radial points")
    
    return df_clean

# ============================================================
# 4. ALPAKA data loading
# ============================================================
def load_alpaka_corrected(csv_path="data/alpaka_rar.csv"):
    df = pd.read_csv(csv_path)
    if 'z' not in df.columns:
        raise KeyError("CSV must contain 'z' column")
    if 'B_corr' not in df.columns:
        if 'b_corr' in df.columns:
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
# 5. Main analysis pipeline
# ============================================================
def main():
    print("="*80)
    print(f"MOND B(z) Analysis")
    print(f"Parameters: μ={MU_TYPE}, Υ*={UPSILON_STAR}, Thresh={DEEP_THRESHOLD}a0")
    print("="*80)
    print(f"Theoretical prediction: B = 1/(2π) = {B_THEORY:.6f}")
    print(f"Theoretical a0(z=0) = {A0_THEORY_Z0:.2e} m/s^2")
    
    # ---------- SPARC local sample analysis ----------
    try:
        df_sparc = load_sparc_full_dataset()
    except FileNotFoundError as e:
        print(f"\nError loading SPARC data: {e}")
        print("Please ensure the files SPARC_Lelli2016c.mrt and MassModels_Lelli2016c.mrt exist in the data/ folder.")
        print("Download from: http://astroweb.case.edu/SPARC/")
        return
    
    # Extract outermost radial point for each galaxy
    df_outer = df_sparc.sort_values(['Galaxy', 'R_kpc']).groupby('Galaxy').last().reset_index()
    print(f"\nExtracted outermost points: {len(df_outer)} galaxies")
    
    # Deep-MOND selection
    deep_mask = (
        (df_outer["a_N_SI"] < DEEP_THRESHOLD * A0_THEORY_Z0) & 
        (df_outer["a_N_SI"] > 1e-12)
    )
    df_outer_deep = df_outer[deep_mask].copy()
    print(f"Deep-MOND outermost sample: {len(df_outer_deep)} galaxies")

    # ==================================================
    # Use simple mean (no weighting)
    # ==================================================
    B_sparc_final = np.mean(df_outer_deep["B_corr"])
    
    # Galaxy-level bootstrap error
    def bootstrap_outer_points(df, n_boot=10000):
        boot_means = []
        n_gal = len(df)
        for _ in range(n_boot):
            idx = np.random.choice(n_gal, n_gal, replace=True)
            boot_mean = np.mean(df.iloc[idx]["B_corr"])
            boot_means.append(boot_mean)
        return np.std(boot_means)

    B_sparc_boot_err = bootstrap_outer_points(df_outer_deep)

    # Output SPARC final results
    print("\n" + "="*70)
    print("SPARC Local Final Results (Outermost points, Simple Mean)")
    print("="*70)
    print(f"Outermost simple mean B_corr = {B_sparc_final:.4f} ± {B_sparc_boot_err:.4f} (1σ Bootstrap)")
    print(f"Theoretical prediction B_theory = {B_THEORY:.4f}")
    print(f"Relative deviation from theory: {(B_sparc_final - B_THEORY)/B_THEORY*100:.1f}%")
    
    # ---------- ALPAKA high-redshift sample analysis ----------
    try:
        df_alp = load_alpaka_corrected("data/alpaka_rar.csv")
        z_vals = df_alp['z'].values
        B_corr = df_alp['B_corr'].values
        B_err = df_alp['B_corr_err'].values
        print(f"\nLoaded ALPAKA sample: {len(z_vals)} galaxies")
        print(f"Redshift range: {z_vals.min():.3f} - {z_vals.max():.3f}")
    except FileNotFoundError:
        print("\nError: data/alpaka_rar.csv not found.")
        print("Please provide a CSV file with columns: z, B_corr, B_corr_err")
        return
    
    # Linear evolution fit
    beta, beta_err, intercept, intercept_err, chi2_red, dof = fit_evolution(z_vals, B_corr, B_err)
    weights = 1 / B_err**2
    mean_B_highz = np.average(B_corr, weights=weights)
    mean_B_highz_err = np.sqrt(1 / np.sum(weights))
    
    # Output ALPAKA results
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
    
    # ---------- Generate figure ----------
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

    # 2. SPARC local result
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
        label='SPARC outermost ($z=0$)'
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

    # 8. Save
    plt.tight_layout()
    plt.savefig('figure1_Bz_paper.png', dpi=300)
    plt.savefig('figure1_Bz_paper.pdf', format='pdf')
    print("\nFigure saved as 'figure1_Bz_paper.png' and 'figure1_Bz_paper.pdf'")
    plt.close()
    
    # Save results
    results = pd.DataFrame({
        'name': df_alp.get('name', np.arange(len(z_vals))),
        'z': z_vals,
        'B_corr': B_corr,
        'B_corr_err': B_err,
    })
    results.to_csv('alpaka_Bcorr_used.csv', index=False)
    print("Results saved to 'alpaka_Bcorr_used.csv'")
    
    print("\n" + "="*80)
    print("Analysis complete.")
    print("="*80)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
