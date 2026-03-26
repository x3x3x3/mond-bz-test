import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set up matplotlib for publication-quality plots
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 14
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 12
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'

# =============================================================================
# Core physical constants and cosmology
# =============================================================================
C = 3e8  # Speed of light in m/s
H0_PLANCK = 67.4  # Planck 2018 H0 in km/s/Mpc
OMEGA_M_PLANCK = 0.311
OMEGA_LAMBDA_PLANCK = 0.689
A0_LOCAL_FIT = 1.2e-10  # Local best-fit a0 in m/s^2
A0_THEORETICAL_Z0 = C * H0_PLANCK * 1000 / (2 * np.pi * 3.086e22)  # ~1.04e-10 m/s^2
BZ_THEORETICAL = 1 / (2 * np.pi)  # ~0.159

def H_z(z, H0=H0_PLANCK, Omega_m=OMEGA_M_PLANCK, Omega_Lambda=OMEGA_LAMBDA_PLANCK):
    """
    Calculate Hubble parameter at redshift z.
    
    Parameters
    ----------
    z : float or array
        Redshift
    H0 : float, optional
        Hubble constant at z=0 in km/s/Mpc
    Omega_m : float, optional
        Matter density parameter
    Omega_Lambda : float, optional
        Dark energy density parameter
    
    Returns
    -------
    H : float or array
        Hubble parameter at redshift z in km/s/Mpc
    """
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)

# =============================================================================
# Core analysis functions
# =============================================================================
def calculate_Bz(a_tot, a_N, z, c=C, H0=H0_PLANCK):
    """
    Calculate the dimensionless observable B(z).
    
    Parameters
    ----------
    a_tot : float or array
        Total centripetal acceleration in m/s^2
    a_N : float or array
        Newtonian acceleration from baryonic matter in m/s^2
    z : float or array
        Redshift
    c : float, optional
        Speed of light in m/s
    H0 : float, optional
        Hubble constant at z=0 in km/s/Mpc
    
    Returns
    -------
    Bz : float or array
        Dimensionless observable B(z)
    """
    # Get H(z) in km/s/Mpc, convert to 1/s
    H_z_val = H_z(z, H0=H0)
    H_z_s = H_z_val * 1000 / 3.086e22
    
    # Calculate B(z)
    Bz = a_tot**2 / (c * H_z_s * a_N)
    
    return Bz

def calculate_Bz_corr(a_tot, a_N, z, mu_function='simple'):
    """
    Calculate the corrected B(z) for non-deep-MOND regimes.
    
    Parameters
    ----------
    a_tot : float or array
        Total centripetal acceleration in m/s^2
    a_N : float or array
        Newtonian acceleration from baryonic matter in m/s^2
    z : float or array
        Redshift
    mu_function : str, optional
        Interpolation function ('simple' or 'standard')
    
    Returns
    -------
    Bz_corr : float or array
        Corrected dimensionless observable B(z)
    """
    # First calculate raw B(z)
    Bz_raw = calculate_Bz(a_tot, a_N, z)
    
    # Calculate a_tot/(c H(z)) term
    H_z_val = H_z(z)
    H_z_s = H_z_val * 1000 / 3.086e22
    term = a_tot / (C * H_z_s)
    
    # Apply correction based on interpolation function
    if mu_function == 'simple':
        # Simple mu(x) = x/(1+x)
        Bz_corr = (Bz_raw - term) / (1 - term)
    else:
        # Standard mu(x) = x/sqrt(1+x^2) - correction derived in paper
        # For simplicity, we use the simple form as in the main paper
        Bz_corr = (Bz_raw - term) / (1 - term)
    
    return Bz_corr

def monte_carlo_error_propagation(a_tot, a_tot_err, a_N, a_N_err, z, z_err=0, n_samples=10000):
    """
    Monte Carlo error propagation for B(z)^corr.
    
    Parameters
    ----------
    a_tot : float
        Total centripetal acceleration in m/s^2
    a_tot_err : float
        Uncertainty in a_tot
    a_N : float
        Newtonian acceleration from baryonic matter in m/s^2
    a_N_err : float
        Uncertainty in a_N
    z : float
        Redshift
    z_err : float, optional
        Uncertainty in redshift (usually negligible)
    n_samples : int, optional
        Number of Monte Carlo samples
    
    Returns
    -------
    Bz_mean : float
        Mean corrected B(z)
    Bz_err : float
        1-sigma uncertainty in corrected B(z)
    """
    # Generate random samples
    a_tot_samples = np.random.normal(a_tot, a_tot_err, n_samples)
    a_N_samples = np.random.normal(a_N, a_N_err, n_samples)
    z_samples = np.random.normal(z, z_err, n_samples)
    
    # Calculate B(z)^corr for each sample
    Bz_samples = calculate_Bz_corr(a_tot_samples, a_N_samples, z_samples)
    
    # Calculate mean and 1-sigma uncertainty
    Bz_mean = np.mean(Bz_samples)
    Bz_err = np.std(Bz_samples)
    
    return Bz_mean, Bz_err

# =============================================================================
# Plotting function (matches Figure 1 in the paper)
# =============================================================================
def plot_Bz_vs_redshift(save_path='image.png'):
    """
    Plot B(z) vs redshift, matching Figure 1 in the paper.
    
    Note: This function uses the published results from the paper.
          To reproduce from raw data, download SPARC and ALPAKA data
          from the sources listed in the README.
    
    Parameters
    ----------
    save_path : str, optional
        Path to save the figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 7))
    
    # Plot theoretical prediction
    ax.axhline(y=BZ_THEORETICAL, color='r', linestyle='--', 
               label=r'Theoretical prediction $\mathcal{B}=1/(2\pi) \approx 0.159$')
    
    # -------------------------------------------------------------------------
    # Local SPARC sample (illustrative, based on paper results)
    # -------------------------------------------------------------------------
    # For full reproducibility, download SPARC data from http://astroweb.cwru.edu/SPARC/
    # Here we show the mean and 1-sigma band as reported in the paper
    local_mean = 0.159
    local_std = 0.025
    ax.axhline(y=local_mean, color='gray', linestyle='-', alpha=0.7)
    ax.fill_between([-0.1, 2.3], local_mean - local_std, local_mean + local_std, 
                    color='gray', alpha=0.2, label=r'Local mean $\pm 1\sigma$')
    
    # -------------------------------------------------------------------------
    # Fiducial high-redshift ALPAKA sample (from Table 1 in the paper)
    # -------------------------------------------------------------------------
    # These are the 5 D-class galaxies from Rizzo et al. (2023) ID1, ID6, ID7, ID12, ID13
    redshifts_highz = np.array([0.561, 1.456, 1.466, 1.634, 2.103])
    Bz_highz = np.array([0.156, 0.149, 0.172, 0.165, 0.141])
    Bz_highz_err = np.array([0.075, 0.072, 0.082, 0.079, 0.070])
    
    # Plot high-redshift points with error bars
    ax.errorbar(redshifts_highz, Bz_highz, yerr=Bz_highz_err, 
                fmt='o', color='b', capsize=3, elinewidth=1.5, 
                label='ALPAKA high-$z$ sample (corrected)')
    
    # -------------------------------------------------------------------------
    # Axes setup
    # -------------------------------------------------------------------------
    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'Dimensionless $\mathcal{B}(z)$')
    ax.set_xlim(-0.1, 2.3)
    ax.set_ylim(0, 0.5)
    ax.legend(loc='upper right', frameon=True)
    
    # Save figure
    plt.savefig(save_path)
    plt.close()
    
    print(f"Figure saved to {save_path}")
    print("Note: For full raw data reproducibility, download SPARC and ALPAKA data")
    print("      from the sources listed in the README.")

# =============================================================================
# Example usage
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("MOND B(z) Analysis Code")
    print("="*60)
    print("\nThis code reproduces the analysis from Xiong (2026, MNRAS Letters submitted).")
    print("\nFor full raw data reproducibility:")
    print("1. Download SPARC data from http://astroweb.cwru.edu/SPARC/")
    print("2. Download ALPAKA data from Rizzo et al. (2023) supplementary material")
    print("\nGenerating Figure 1 (based on published results)...")
    plot_Bz_vs_redshift()
    print("\nDone.")
