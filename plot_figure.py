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

if __name__ == "__main__":
    # Example usage
    print("MOND B(z) plotting code")
    print("------------------------")
    
    # Example data
    # Local sample (synthetic data for example)
    n_local = 175
    redshifts_local = np.zeros(n_local)
    Bz_local = np.random.normal(0.159, 0.025, n_local)
    Bz_local_err = np.random.uniform(0.01, 0.03, n_local)
    
    # High-redshift sample (fiducial 5 galaxies)
    redshifts_highz = np.array([0.561, 1.456, 1.466, 1.634, 2.103])
    Bz_highz = np.array([0.156, 0.149, 0.172, 0.165, 0.141])
    Bz_highz_err = np.array([0.075, 0.072, 0.082, 0.079, 0.070])
    
    # Plot figure
    plot_Bz_vs_redshift(redshifts_local, Bz_local, Bz_local_err,
                        redshifts_highz, Bz_highz, Bz_highz_err)
