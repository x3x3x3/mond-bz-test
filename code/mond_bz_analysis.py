#!/usr/bin/env python3
"""
plot_Bz.py
Generate Figure 1 (B(z) vs redshift) from SPARC and ALPAKA data.
Data files (MassModels_Lelli2016c.mrt and alpaka_rar.csv) must be in the ../data/ directory.
alpaka_rar.csv must contain the columns 'B_corr' and 'B_corr_err'.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# ==================== Path settings ====================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')

sparc_mrt = os.path.join(data_dir, 'MassModels_Lelli2016c.mrt')
alpaka_csv = os.path.join(data_dir, 'alpaka_rar.csv')
sparc_csv = os.path.join(data_dir, 'sparc_rar.csv')

# ==================== Generate sparc_rar.csv if missing ====================
def generate_sparc_rar():
    print("Generating sparc_rar.csv from MassModels_Lelli2016c.mrt ...")
    try:
        kpc_to_m = 3.086e19
        kms_to_ms = 1000.0

        # Read Table2.mrt (skip 29 header lines, adjust if needed)
        df = pd.read_csv(sparc_mrt, delim_whitespace=True, comment='#',
                         names=['ID', 'D', 'R', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul'],
                         skiprows=29)

        idx = df.groupby('ID')['R'].idxmax()
        df_outer = df.loc[idx].copy()

        R_m = df_outer['R'].values * kpc_to_m
        Vobs_ms = df_outer['Vobs'].values * kms_to_ms
        Vobs_err_ms = df_outer['e_Vobs'].values * kms_to_ms

        Vgas_ms = df_outer['Vgas'].values * kms_to_ms
        Vdisk_ms = df_outer['Vdisk'].values * kms_to_ms
        Vbul_ms = df_outer['Vbul'].values * kms_to_ms
        Vbar_ms = np.sqrt(Vgas_ms**2 + Vdisk_ms**2 + Vbul_ms**2)
        rel_err = Vobs_err_ms / Vobs_ms
        Vbar_err_ms = Vbar_ms * rel_err

        g_obs = Vobs_ms**2 / R_m
        g_obs_err = 2 * Vobs_err_ms / Vobs_ms * g_obs
        g_bar = Vbar_ms**2 / R_m
        g_bar_err = 2 * Vbar_err_ms / Vbar_ms * g_bar

        out = pd.DataFrame({
            'name': df_outer['ID'].values,
            'z': 0.0,
            'g_bar': g_bar / 1e-10,
            'g_obs': g_obs / 1e-10,
            'g_bar_err': g_bar_err / 1e-10,
            'g_obs_err': g_obs_err / 1e-10
        })
        out.to_csv(sparc_csv, index=False)
        print(f"Created {sparc_csv} with {len(out)} galaxies.")
        return True
    except Exception as e:
        print(f"Failed to generate sparc_rar.csv: {e}")
        return False

if not os.path.exists(sparc_csv):
    print("sparc_rar.csv not found. Generating it now...")
    if not generate_sparc_rar():
        sys.exit(1)
else:
    print("sparc_rar.csv found. Using it directly.")

if not os.path.exists(alpaka_csv):
    print(f"Error: High-redshift data file not found at {alpaka_csv}")
    sys.exit(1)

# ==================== Load data ====================
sparc = pd.read_csv(sparc_csv)
alpaka = pd.read_csv(alpaka_csv)

# Check that the required columns exist
if 'B_corr' not in alpaka.columns or 'B_corr_err' not in alpaka.columns:
    print("Error: alpaka_rar.csv must contain 'B_corr' and 'B_corr_err' columns.")
    sys.exit(1)

# ==================== Compute B(0) for SPARC ====================
c = 3e8                     # m/s
H0_km_s_Mpc = 67.4          # Planck 2018
H0_s = H0_km_s_Mpc * 1000 / 3.086e22   # s^-1

def compute_B0(g_obs_1e10, g_bar_1e10):
    a_tot = g_obs_1e10 * 1e-10
    a_N   = g_bar_1e10 * 1e-10
    return (a_tot**2) / (c * H0_s * a_N)

sparc['B'] = compute_B0(sparc['g_obs'], sparc['g_bar'])
sparc['B_err'] = sparc['B'] * np.sqrt((sparc['g_obs_err']/sparc['g_obs'])**2 +
                                      (sparc['g_bar_err']/sparc['g_bar'])**2)

# ==================== Select high-redshift sample ====================
# To use only the 5 fiducial galaxies (IDs: 1,6,7,12,13) from the Letter,
# uncomment the next lines and adjust the names accordingly.
# fiducial_names = ['GOODS-S_15503', 'ALMA.08', 'ALMA.01', 'SpARCS_J0224-159', 'COSMOS_3182']
# alpaka = alpaka[alpaka['name'].isin(fiducial_names)]

# ==================== Plot ====================
theory = 1 / (2 * np.pi)   # ≈ 0.159

fig, ax = plt.subplots(figsize=(8, 6))

# Theoretical line
ax.axhline(y=theory, color='red', linestyle='--', linewidth=2,
           label=r'Theoretical prediction $\mathcal{B}=1/(2\pi) \approx 0.159$')

# SPARC local galaxies
ax.scatter(sparc['z'], sparc['B'], color='gray', alpha=0.3, s=10,
           label='SPARC $z=0$ sample')

# Local mean ±1σ band
local_mean = sparc['B'].mean()
local_std = sparc['B'].std()
ax.axhline(y=local_mean, color='gray', linestyle='-', alpha=0.7)
ax.fill_between([-0.1, 2.5], local_mean - local_std, local_mean + local_std,
                color='gray', alpha=0.2, label=r'Local mean $\pm 1\sigma$')

# High-redshift points (corrected)
ax.errorbar(alpaka['z'], alpaka['B_corr'], yerr=alpaka['B_corr_err'],
            fmt='o', color='blue', capsize=3, elinewidth=1.5,
            label='ALPAKA high-$z$ sample (corrected)')

ax.set_xlabel(r'Redshift $z$')
ax.set_ylabel(r'$\mathcal{B}(z)$')
ax.set_xlim(-0.1, 2.5)
ax.set_ylim(0, 0.5)
ax.legend(loc='upper right')
plt.tight_layout()

output_path = os.path.join(script_dir, 'Bz_plot.png')
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Figure saved to {output_path}")
