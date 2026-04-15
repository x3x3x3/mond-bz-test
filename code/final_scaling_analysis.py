#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINAL COMPLETE CODE: Cosmic Scaling Analysis with All Corrections
- Shuffle test: only y permuted, x fixed.
- H0 statistics: reports both unfiltered and filtered medians with MAD.
- HiZELS labeled as illustrative (z=0 a_z only).
- Generates: figure1 (collapse with insets), figure2 (shuffle test),
  supp_combined.pdf (2x2: environment, H0 distribution, HiZELS illustration).
"""

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic, spearmanr, norm
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize

# ============================================================
# Physical Constants (SI)
# ============================================================
C = 2.99792458e8
KPC_TO_M = 3.085677581e19
KM_TO_M = 1e3
UPSILON = 0.6
H0_PLANCK = 67.4
H0_SI = H0_PLANCK * 1e3 / 3.085677581e22
A_Z_REF = C * H0_SI / (2 * np.pi)        # ~1.042e-10 m/s²

def clean_name(name):
    return re.sub(r'[^a-zA-Z0-9]', '', str(name)).lower()

# ---------- Parse Table1 ----------
def parse_table1(filepath="data/SPARC/table1.dat"):
    galaxies = {}
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 18:
                continue
            try:
                name_raw = parts[0]
                inc = float(parts[5])
                L36 = float(parts[7])
                MHI = float(parts[13])
                Q = int(parts[17])
                if Q <= 2 and inc >= 20:
                    galaxies[clean_name(name_raw)] = {
                        'name': name_raw,
                        'inc': inc,
                        'L36': L36,
                        'MHI': MHI
                    }
            except (ValueError, IndexError):
                continue
    print(f"Loaded {len(galaxies)} high-quality galaxies from Table1")
    return galaxies

# ---------- Parse Table2 ----------
def parse_table2(filepath="data/SPARC/table2.dat"):
    data_dict = {}
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                name_raw = parts[0]
                R = float(parts[2])
                Vobs = float(parts[3])
                eVobs = float(parts[4]) if len(parts) > 4 else 0.1 * Vobs
                Vgas = float(parts[5]) if len(parts) > 5 else 0.0
                Vdisk = float(parts[6]) if len(parts) > 6 else 0.0
                Vbul = float(parts[7]) if len(parts) > 7 else 0.0
                name_clean = clean_name(name_raw)
                if name_clean not in data_dict:
                    data_dict[name_clean] = []
                data_dict[name_clean].append({
                    'R_kpc': R, 'Vobs': Vobs, 'eVobs': eVobs,
                    'Vgas': Vgas, 'Vdisk': Vdisk, 'Vbul': Vbul
                })
            except ValueError:
                continue
    return data_dict

# ---------- Compute all points ----------
def compute_all_points():
    galaxies = parse_table1()
    rot_data = parse_table2()
    points = []
    for name_clean, props in galaxies.items():
        if name_clean not in rot_data:
            continue
        inc = props['inc']
        sin_i = np.sin(np.radians(inc))
        for p in rot_data[name_clean]:
            R_m = p['R_kpc'] * KPC_TO_M
            Vobs_corr = p['Vobs'] / sin_i * KM_TO_M
            Vgas = p['Vgas'] * KM_TO_M
            Vdisk = p['Vdisk'] * KM_TO_M * np.sqrt(UPSILON)
            Vbul = p['Vbul'] * KM_TO_M * np.sqrt(UPSILON)
            Vbar2 = Vgas**2 + Vdisk**2 + Vbul**2
            a_tot = Vobs_corr**2 / R_m
            a_N = Vbar2 / R_m
            if a_N <= 0 or a_tot <= 0:
                continue
            x = a_N / A_Z_REF
            y = a_tot / A_Z_REF
            points.append({
                'name': props['name'],
                'x': x, 'y': y,
                'a_tot': a_tot, 'a_N': a_N,
                'R_kpc': p['R_kpc']
            })
    df = pd.DataFrame(points)
    print(f"Total data points: {len(df)}")
    return df

# ---------- Run computation ----------
df = compute_all_points()
x = df['x'].values
y = df['y'].values
logx = np.log10(x)
logy = np.log10(y)

# ---------- Bootstrap median error ----------
def bootstrap_median(data, n_boot=10000):
    rng = np.random.default_rng(42)
    meds = [np.median(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    return np.std(meds)

med_x = np.median(x)
med_x_err = bootstrap_median(x)
print(f"Median x = {med_x:.3f} ± {med_x_err:.3f}")
print(f"Fraction x < 1: {np.mean(x < 1)*100:.1f}%")
print(f"Std dev of log10(x): {np.std(logx):.3f} dex")

# ---------- Spearman correlation ----------
rho, pval = spearmanr(logx, logy)
print(f"Spearman ρ = {rho:.4f}, p = {pval:.4e}")

# ---------- Binned statistics ----------
bins = np.logspace(np.log10(x.min()), np.log10(x.max()), 20)
x_bin = np.sqrt(bins[:-1] * bins[1:])
y_med, _, _ = binned_statistic(x, y, statistic='median', bins=bins)
y_low, _, _ = binned_statistic(x, y, statistic=lambda v: np.percentile(v, 16), bins=bins)
y_high, _, _ = binned_statistic(x, y, statistic=lambda v: np.percentile(v, 84), bins=bins)

# ---------- Residuals ----------
mask = np.isfinite(x_bin) & np.isfinite(y_med)
spl = UnivariateSpline(np.log10(x_bin[mask]), np.log10(y_med[mask]), s=0.5)
logy_pred = spl(logx)
residuals = logy - logy_pred
mu_res, std_res = norm.fit(residuals)

# ---------- CORRECTED Shuffle test (only y permuted) ----------
n_shuffle = 10000
rho_shuffle = []
rng = np.random.default_rng(42)
for _ in range(n_shuffle):
    y_shuffled = rng.permutation(y)
    rho_s, _ = spearmanr(logx, np.log10(y_shuffled))
    rho_shuffle.append(rho_s)
rho_shuffle = np.array(rho_shuffle)
p_shuffle = np.mean(rho_shuffle >= rho)
print(f"Corrected Shuffle test p-value: {p_shuffle:.4e}")

# ============================================================
# FIGURE 1: Scaling collapse with insets
# ============================================================
fig = plt.figure(figsize=(8, 7))
ax_main = fig.add_subplot(111)
ax_main.set_xscale('log')
ax_main.set_yscale('log')

ax_main.loglog(x, y, 'o', markersize=1, alpha=0.25, color='gray', rasterized=True)
ax_main.errorbar(x_bin, y_med, yerr=[y_med - y_low, y_high - y_med],
                 fmt='s', color='#1f77b4', markersize=5, capsize=2,
                 label='SPARC binned median')
x_theory = np.logspace(-2, 2, 100)
y_theory = (x_theory + np.sqrt(x_theory**2 + 4*x_theory)) / 2
ax_main.loglog(x_theory, y_theory, 'r--', lw=1.5, alpha=0.8,
               label='MOND ($a_0=a_z$; not a fit)')
ax_main.set_xlabel('$x = a_N / a_z$')
ax_main.set_ylabel('$y = a_{\\rm tot} / a_z$')
ax_main.set_xlim(3e-2, 3e1)
ax_main.set_ylim(1e-2, 1e2)
ax_main.legend(loc='upper left', frameon=False)
ax_main.text(0.05, 0.95, '(a)', transform=ax_main.transAxes, fontsize=12, va='top')

# Inset 1: x distribution
ax_inset1 = ax_main.inset_axes([0.62, 0.62, 0.25, 0.25])
ax_inset1.hist(logx, bins=40, alpha=0.7, color='#2ca02c', edgecolor='white')
ax_inset1.axvline(0, color='k', linestyle='--', lw=1)
ax_inset1.set_xlabel('$\log_{10} x$', fontsize=8)
ax_inset1.set_ylabel('Count', fontsize=8)
ax_inset1.tick_params(labelsize=7)
ax_inset1.text(0.05, 0.95, f'median = {med_x:.2f}', transform=ax_inset1.transAxes,
               fontsize=7, va='top')

# Inset 2: Residuals
ax_inset2 = ax_main.inset_axes([0.62, 0.15, 0.25, 0.25])
ax_inset2.hist(residuals, bins=30, density=True, alpha=0.7, color='#ff7f0e', edgecolor='white')
x_plot = np.linspace(-0.4, 0.4, 100)
ax_inset2.plot(x_plot, norm.pdf(x_plot, mu_res, std_res), 'k-', lw=1)
ax_inset2.set_xlabel('$\Delta \log y$', fontsize=8)
ax_inset2.set_ylabel('Density', fontsize=8)
ax_inset2.tick_params(labelsize=7)
ax_inset2.text(0.05, 0.95, f'$\sigma = {std_res:.2f}$ dex', transform=ax_inset2.transAxes,
               fontsize=7, va='top')

plt.subplots_adjust(right=0.85, left=0.10)
plt.savefig('figure1_collapse_with_insets.pdf', dpi=300)
print("Saved: figure1_collapse_with_insets.pdf")

# ============================================================
# FIGURE 2: Corrected Shuffle test
# ============================================================
plt.figure(figsize=(8, 5))
plt.hist(rho_shuffle, bins=50, alpha=0.7, color='#9467bd', edgecolor='white')
plt.axvline(rho, color='k', linestyle='--', lw=2, label=f'Observed $\\rho = {rho:.3f}$')
plt.xlabel('Spearman $\\rho$ (y shuffled)')
plt.ylabel('Frequency')
plt.legend()
plt.title(f'Shuffle Test: $p = {p_shuffle:.2e}$')
plt.tight_layout()
plt.savefig('figure2_shuffle_test.pdf', dpi=300)
print("Saved: figure2_shuffle_test.pdf")

# ============================================================
# Per-galaxy az fitting (H0 distribution)
# ============================================================
print("\nPer-galaxy az fitting...")
def global_relation(x_val):
    return 10**spl(np.log10(x_val))

h0_fits = []
az_fits = []
for name in df['name'].unique():
    sub = df[df['name'] == name]
    if len(sub) < 5:
        continue
    x_sub = sub['x'].values
    y_sub = sub['y'].values
    def cost(log_az):
        az = np.exp(log_az)
        x_scaled = x_sub * (A_Z_REF / az)
        y_scaled = y_sub * (A_Z_REF / az)
        y_pred = global_relation(x_scaled)
        return np.sum((np.log10(y_scaled) - np.log10(y_pred))**2)
    res = minimize(cost, x0=np.log(A_Z_REF), method='Nelder-Mead')
    az_best = np.exp(res.x[0])
    h0_best = 2 * np.pi * az_best / C * 3.085677581e19
    az_fits.append(az_best)
    h0_fits.append(h0_best)

az_fits = np.array(az_fits)
h0_fits = np.array(h0_fits)

# Unfiltered statistics
print(f"Unfiltered median H0 = {np.median(h0_fits):.1f} km/s/Mpc")
print(f"Unfiltered MAD = {np.median(np.abs(h0_fits - np.median(h0_fits))):.1f} km/s/Mpc")

# Filter outliers: az within factor 5 of reference
mask_valid = (az_fits > A_Z_REF / 5) & (az_fits < A_Z_REF * 5)
h0_valid = h0_fits[mask_valid]
print(f"Valid H0 fits: {len(h0_valid)}/{len(h0_fits)}")
print(f"Median H0 (filtered) = {np.median(h0_valid):.1f} km/s/Mpc")
print(f"MAD (filtered) = {np.median(np.abs(h0_valid - np.median(h0_valid))):.1f} km/s/Mpc")

# ============================================================
# HiZELS loading (illustrative only)
# ============================================================
def load_hizels():
    try:
        prop = pd.read_csv('data/hizels_properties.csv')
        kin = pd.read_csv('data/hizels_kinematics.csv')
    except FileNotFoundError:
        print("HiZELS files not found. Skipping.")
        return pd.DataFrame()
    df_h = pd.merge(prop, kin, on='Name', how='inner')
    points = []
    for _, row in df_h.iterrows():
        R_m = row['Rh'] * KPC_TO_M
        V = row['Vrot'] * KM_TO_M
        Mstar = 10**row['logMass'] * 1.9885e30
        a_tot = V**2 / R_m
        a_N = 6.6743e-11 * Mstar / R_m**2
        if a_N > 0 and a_tot > 0:
            points.append({
                'name': row['Name'],
                'z': row['z'],
                'x': a_N / A_Z_REF,
                'y': a_tot / A_Z_REF
            })
    return pd.DataFrame(points)

print("\nLoading HiZELS...")
df_hiz = load_hizels()
if not df_hiz.empty:
    print(f"HiZELS: {len(df_hiz)} galaxies (illustrative only)")

# ============================================================
# SUPPLEMENTARY: Environment dependence + H0 hist + HiZELS
# ============================================================
props = parse_table1()
gal_med = df.groupby('name')['x'].median().reset_index()
gal_med.columns = ['name', 'x_med']
gal_data = []
for _, row in gal_med.iterrows():
    name_clean = clean_name(row['name'])
    if name_clean in props:
        p = props[name_clean]
        Mstar = UPSILON * p['L36'] * 1e9
        Mgas = 1.33 * p['MHI'] * 1e9
        gal_data.append({
            'x_med': row['x_med'],
            'Mstar': Mstar,
            'f_gas': Mgas/(Mstar+Mgas) if (Mstar+Mgas)>0 else np.nan
        })
df_gal = pd.DataFrame(gal_data).dropna()

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
# Top-left: Mstar
ax = axes[0,0]
ax.scatter(df_gal['Mstar'], df_gal['x_med'], alpha=0.6, s=10, color='#d62728')
ax.set_xscale('log')
ax.set_xlabel('$M_*$ [$M_\\odot$]')
ax.set_ylabel('Median $x$')
ax.axhline(1.0, color='k', linestyle='--', lw=1)
ax.set_title('Stellar Mass')
# Top-right: f_gas
ax = axes[0,1]
ax.scatter(df_gal['f_gas'], df_gal['x_med'], alpha=0.6, s=10, color='#9467bd')
ax.set_xlabel('$f_{\\rm gas}$')
ax.set_ylabel('Median $x$')
ax.axhline(1.0, color='k', linestyle='--', lw=1)
ax.set_title('Gas Fraction')
# Bottom-left: H0 distribution (filtered)
ax = axes[1,0]
ax.hist(h0_valid, bins=20, alpha=0.7, color='#17becf', edgecolor='white')
ax.axvline(H0_PLANCK, color='k', linestyle='--', lw=2, label=f'Planck $H_0$={H0_PLANCK}')
ax.axvline(np.median(h0_valid), color='#d62728', linestyle='-', lw=2,
           label=f'Median={np.median(h0_valid):.1f}')
ax.set_xlabel('$H_0^{\\rm dyn}$ [km/s/Mpc]')
ax.set_ylabel('Count')
ax.legend(fontsize=8)
ax.set_title('Inferred $H_0$ per Galaxy (filtered)')
# Bottom-right: HiZELS comparison (illustrative)
ax = axes[1,1]
ax.loglog(x, y, 'o', markersize=1, alpha=0.15, color='gray', rasterized=True)
ax.errorbar(x_bin, y_med, yerr=[y_med - y_low, y_high - y_med],
            fmt='s', color='#1f77b4', markersize=4, capsize=2)
ax.loglog(x_theory, y_theory, 'r--', lw=1.5, alpha=0.8)
if not df_hiz.empty:
    ax.scatter(df_hiz['x'], df_hiz['y'], marker='^', s=25, color='#d62728',
               edgecolor='white', linewidth=0.5)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('$x = a_N / a_z$'); ax.set_ylabel('$y = a_{\\rm tot} / a_z$')
ax.set_title('HiZELS (illustrative, $z=0$ $a_z$)')
ax.legend(['SPARC', 'MOND', 'HiZELS'], fontsize=8) if not df_hiz.empty else ax.legend(['SPARC', 'MOND'], fontsize=8)
plt.tight_layout()
plt.savefig('supp_combined.pdf', dpi=300)
print("Saved: supp_combined.pdf")

# ============================================================
# Save data
# ============================================================
df.to_csv('sparc_scaling_data.csv', index=False)
print("Saved: sparc_scaling_data.csv")
print("\nAll tasks completed successfully.")
