
import numpy as np
import pandas as pd
from scipy import stats
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

# ==================== DATA ANALYSIS FUNCTIONS ====================
def calculate_Bz(a_tot, a_N, H_z, c=3e8):
    """
    Calculate the dimensionless observable B(z)
    
    Parameters
    ----------
    a_tot : float or array
        Total centripetal acceleration (m/s^2)
    a_N : float or array
        Newtonian acceleration from baryonic matter (m/s^2)
    H_z : float or array
        Hubble parameter at redshift z (km/s/Mpc)
    c : float, optional
        Speed of light (m/s), default 3e8
    
    Returns
    -------
    Bz : float or array
        Dimensionless observable B(z)
    """
    # Convert H_z from km/s/Mpc to 1/s
    H_z_s = H_z * 1000 / (3.086e22)
    
    # Calculate B(z)
    Bz = a_tot**2 / (c * H_z_s * a_N)
    
    return Bz

def monte_carlo_error_propagation(a_tot, a_tot_err, a_N, a_N_err, H_z, H_z_err, n_samples=10000):
    """
    Monte Carlo error propagation for B(z)
    
    Parameters
    ----------
    a_tot : float
        Total centripetal acceleration (m/s^2)
    a_tot_err : float
        Uncertainty in a_tot
    a_N : float
        Newtonian acceleration from baryonic matter (m/s^2)
    a_N_err : float
        Uncertainty in a_N
    H_z : float
        Hubble parameter at redshift z (km/s/Mpc)
    H_z_err : float
        Uncertainty in H_z
    n_samples : int, optional
        Number of Monte Carlo samples, default 10000
    
    Returns
    -------
    Bz_mean : float
        Mean B(z)
    Bz_err : float
        1-sigma uncertainty in B(z)
    """
    # Generate random samples
    a_tot_samples = np.random.normal(a_tot, a_tot_err, n_samples)
    a_N_samples = np.random.normal(a_N, a_N_err, n_samples)
    H_z_samples = np.random.normal(H_z, H_z_err, n_samples)
    
    # Calculate B(z) for each sample
    Bz_samples = calculate_Bz(a_tot_samples, a_N_samples, H_z_samples)
    
    # Calculate mean and 1-sigma uncertainty
    Bz_mean = np.mean(Bz_samples)
    Bz_err = np.std(Bz_samples)
    
    return Bz_mean, Bz_err

def randomization_null_test(Bz_values, redshifts, n_permutations=10000):
    """
    Randomization null test to check if B(z) is consistent with a constant
    
    Parameters
    ----------
    Bz_values : array
        Measured B(z) values
    redshifts : array
        Redshifts of the measurements
    n_permutations : int, optional
        Number of permutations, default 10000
    
    Returns
    -------
    p_value : float
        p-value for the null test
    """
    # Calculate the observed slope
    slope_obs, intercept_obs, r_value_obs, p_value_obs, std_err_obs = stats.linregress(redshifts, Bz_values)
    
    # Perform permutations
    slope_perm = np.zeros(n_permutations)
    for i in range(n_permutations):
        # Shuffle the B(z) values
        Bz_shuffled = np.random.permutation(Bz_values)
        
        # Calculate the slope for the shuffled data
        slope_perm[i], _, _, _, _ = stats.linregress(redshifts, Bz_shuffled)
    
    # Calculate the p-value
    p_value = np.sum(np.abs(slope_perm) >= np.abs(slope_obs)) / n_permutations
    
    return p_value

# ==================== PLOTTING FUNCTIONS ====================
def plot_Bz_vs_redshift(redshifts_local, Bz_local, Bz_local_err,
                        redshifts_highz, Bz_highz, Bz_highz_err,
                        theoretical_prediction=1/(2*np.pi),
                        save_path='image.png'):
    """
    Plot B(z) vs redshift
    
    Parameters
    ----------
    redshifts_local : array
        Redshifts of local galaxies
    Bz_local : array
        B(z) values of local galaxies
    Bz_local_err : array
        Uncertainties in B(z) of local galaxies
    redshifts_highz : array
        Redshifts of high-redshift galaxies
    Bz_highz : array
        B(z) values of high-redshift galaxies
    Bz_highz_err : array
        Uncertainties in B(z) of high-redshift galaxies
    theoretical_prediction : float, optional
        Theoretical prediction for B(z), default 1/(2*pi)
    save_path : str, optional
        Path to save the figure, default 'image.png'
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 7))
    
    # Plot theoretical prediction
    ax.axhline(y=theoretical_prediction, color='r', linestyle='--', 
               label=r'Theoretical prediction $\mathcal{B}=1/(2\pi) \approx 0.159$')
    
    # Plot local sample (semi-transparent to avoid overcrowding)
    ax.scatter(redshifts_local, Bz_local, color='gray', alpha=0.3, s=10,
               label='SPARC $z=0$ sample')
    
    # Plot local mean and 1-sigma band
    local_mean = np.mean(Bz_local)
    local_std = np.std(Bz_local)
    ax.axhline(y=local_mean, color='gray', linestyle='-', alpha=0.7)
    ax.fill_between([-0.1, 2.3], local_mean - local_std, local_mean + local_std, 
                    color='gray', alpha=0.2, label=r'Local mean $\pm 1\sigma$')
    
    # Plot high-redshift sample with error bars
    ax.errorbar(redshifts_highz, Bz_highz, yerr=Bz_highz_err, 
                fmt='o', color='b', capsize=3, elinewidth=1.5, 
                label='ALPAKA high-$z$ sample (corrected)')
    
    # Set axes labels and title
    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'Dimensionless $\mathcal{B}(z)$')
    
    # Set axes limits
    ax.set_xlim(-0.1, 2.3)
    ax.set_ylim(0, 0.5)
    
    # Add legend
    ax.legend(loc='upper right', frameon=True)
    
    # Save figure
    plt.savefig(save_path)
    plt.close()
    
    print(f"Figure saved to {save_path}")

# ==================== MAIN EXAMPLE ====================
if __name__ == "__main__":
    print("="*50)
    print("MOND B(z) Analysis and Plotting Code")
    print("="*50)
    
    # Example data
    print("\n1. Running example analysis...")
    
    # Local sample (synthetic data for example)
    n_local = 175
    redshifts_local = np.zeros(n_local)
    Bz_local = np.random.normal(0.159, 0.025, n_local)
    Bz_local_err = np.random.uniform(0.01, 0.03, n_local)
    
    # High-redshift sample (fiducial 5 galaxies)
    redshifts_highz = np.array([0.561, 1.456, 1.466, 1.634, 2.103])
    Bz_highz = np.array([0.156, 0.149, 0.172, 0.165, 0.141])
    Bz_highz_err = np.array([0.075, 0.072, 0.082, 0.079, 0.070])
    
    # Example: Calculate B(z) for one high-redshift galaxy
    a_tot_example = 1.49e-10  # m/s^2
    a_tot_err_example = 0.22e-10  # m/s^2
    a_N_example = 1.0e-10  # m/s^2
    a_N_err_example = 0.2e-10  # m/s^2
    H_z_example = 100  # km/s/Mpc
    H_z_err_example = 5  # km/s/Mpc
    
    Bz_mean, Bz_err = monte_carlo_error_propagation(a_tot_example, a_tot_err_example, 
                                                      a_N_example, a_N_err_example, 
                                                      H_z_example, H_z_err_example)
    print(f"   Example B(z) = {Bz_mean:.3f} ± {Bz_err:.3f}")
    
    # Example: Randomization null test
    print("\n2. Running randomization null test...")
    p_value = randomization_null_test(np.concatenate([Bz_local[:5], Bz_highz]), 
                                      np.concatenate([redshifts_local[:5], redshifts_highz]))
    print(f"   Null test p-value = {p_value:.3f}")
    
    # Example: Plot figure
    print("\n3. Generating plot...")
    plot_Bz_vs_redshift(redshifts_local, Bz_local, Bz_local_err,
                        redshifts_highz, Bz_highz, Bz_highz_err)
    
    print("\n" + "="*50)
    print("Analysis complete!")
    print("="*50)
