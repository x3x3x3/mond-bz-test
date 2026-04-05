"""
MOND B(z) analysis.
Generates Figure 1 and statistical output for MNRAS Letter.
Author: Wenhao Xiong
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==================================================
# Constants
# ==================================================
B_THEORY = 1 / (2 * np.pi)          # 0.1591549430918953

B_SPARC_OUTER = 0.1619
B_SPARC_OUTER_ERR = 0.0099
B_SPARC_DEEP = 0.1585
B_SPARC_DEEP_ERR = 0.0012

# ==================================================
# Load ALPAKA data
# ==================================================
csv_path = "data/alpaka_rar.csv"
try:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    z_col = None
    b_col = None
    b_err_col = None
    for col in df.columns:
        low = col.lower()
        if low == 'z':
            z_col = col
        elif 'b_corr' in low and 'err' not in low:
            b_col = col
        elif 'b_corr_err' in low or 'b_err' in low:
            b_err_col = col
    if z_col is None or b_col is None or b_err_col is None:
        raise KeyError("CSV must contain z, B_corr, B_corr_err columns")
    z_vals = df[z_col].values
    B_corr = df[b_col].values
    B_err = df[b_err_col].values
    print(f"Using columns: z='{z_col}', B_corr='{b_col}', B_err='{b_err_col}'")
    print(f"Redshift range: {z_vals.min():.3f} - {z_vals.max():.3f}")
except Exception as e:
    print(f"Error loading ALPAKA data: {e}")
    exit(1)

# ==================================================
# Linear fit: B(z) = beta * z + intercept
# ==================================================
def linear_model(z, beta, intercept):
    return beta * z + intercept

weights = 1 / B_err**2
popt, pcov = curve_fit(linear_model, z_vals, B_corr, sigma=B_err, absolute_sigma=True)
beta, intercept = popt
beta_err, intercept_err = np.sqrt(np.diag(pcov))

residuals = B_corr - linear_model(z_vals, beta, intercept)
chi2 = np.sum((residuals / B_err)**2)
dof = len(z_vals) - 2
chi2_red = chi2 / dof

mean_B_highz = np.average(B_corr, weights=weights)
mean_B_highz_err = np.sqrt(1 / np.sum(weights))

# ==================================================
# Print results (for paper or console)
# ==================================================
print("\n" + "="*70)
print("ALPAKA High-redshift Results")
print("="*70)
print(f"Number of galaxies: {len(z_vals)}")
print(f"Weighted mean B_corr = {mean_B_highz:.4f} ± {mean_B_highz_err:.4f}")
print(f"Deviation from theory (1/(2π)={B_THEORY:.4f}): {(mean_B_highz - B_THEORY)/B_THEORY*100:.1f}%")
print("\n Linear fit: B(z) = β z + B₀")
print(f"  β = {beta:.4f} ± {beta_err:.4f}  (1σ)")
print(f"  B₀ = {intercept:.4f} ± {intercept_err:.4f}")
print(f"  χ²_red = {chi2_red:.2f} (dof = {dof})")
print(f"  Significance of non-zero slope: {abs(beta/beta_err):.1f}σ")
print("="*70)

# ==================================================
# Generate Figure 1
# ==================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'figure.figsize': (7.5, 5)
})

fig, ax = plt.subplots()

# Theory line
ax.axhline(y=B_THEORY, color='#d62728', linestyle='--', linewidth=2, zorder=1,
           label=r'$\mathcal{B}_{\rm theory}=1/(2\pi)$')

# Best-fit line (drawn first to be in background)
z_fit = np.linspace(-0.1, 2.5, 100)
ax.plot(z_fit, linear_model(z_fit, beta, intercept), 'b--', alpha=0.6, linewidth=1.5, zorder=2,
        label=f'linear fit: $\\beta = {beta:.3f}\\pm{beta_err:.3f}$')

# SPARC points at z=0 (slightly offset horizontally)
ax.scatter(-0.02, B_SPARC_OUTER, marker='*', s=500, color='black', 
           edgecolor='white', linewidth=1, zorder=10, label='SPARC outermost (z=0)')
ax.errorbar(-0.02, B_SPARC_OUTER, yerr=B_SPARC_OUTER_ERR, fmt='none', 
            color='black', capsize=5, capthick=1.5, elinewidth=1.5, zorder=9)

ax.scatter(0.02, B_SPARC_DEEP, marker='s', s=150, color='gray', 
           edgecolor='white', linewidth=1, zorder=10, label='SPARC deep-MOND (z=0)')
ax.errorbar(0.02, B_SPARC_DEEP, yerr=B_SPARC_DEEP_ERR, fmt='none', 
            color='gray', capsize=5, capthick=1.5, elinewidth=1.5, zorder=9)

# ALPAKA points
ax.scatter(z_vals, B_corr, s=60, color='#1f77b4', zorder=5, label='ALPAKA high‑$z$ (corrected)')
ax.errorbar(z_vals, B_corr, yerr=B_err, fmt='none', color='#1f77b4', 
            capsize=3, capthick=1.5, elinewidth=1.5, alpha=0.9, zorder=4)

# Axes
ax.set_xlabel('Redshift $z$', fontsize=12)
ax.set_ylabel(r'$\mathcal{B}(z)$', fontsize=12)
ax.set_xlim(-0.1, 2.4)
ax.set_ylim(0.0, 0.4)
ax.set_xticks(np.arange(0, 2.5, 0.5))
ax.set_yticks(np.arange(0, 0.41, 0.1))
ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray')
ax.grid(True, linestyle=':', alpha=0.4, zorder=0)

# Statistics text box
stats_text = (f"$\\langle \\mathcal{{B}} \\rangle_{{high-z}} = {mean_B_highz:.3f}\\pm{mean_B_highz_err:.3f}$\n"
              f"$\\beta = {beta:.3f}\\pm{beta_err:.3f}$\n"
              f"$\\chi^2_\\mathrm{{red}} = {chi2_red:.2f}$")
ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), zorder=20)

plt.tight_layout()
plt.savefig('figure1_Bz.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_Bz.pdf', bbox_inches='tight')
print("\nFigure saved as 'figure1_Bz.png' and 'figure1_Bz.pdf'")
plt.close()

# ==================================================
# Optionally save used data
# ==================================================
df_out = pd.DataFrame({'z': z_vals, 'B_corr': B_corr, 'B_corr_err': B_err})
df_out.to_csv('alpaka_Bcorr_used.csv', index=False)
print("ALPAKA data saved to 'alpaka_Bcorr_used.csv'")
