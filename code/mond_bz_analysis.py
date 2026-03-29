import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Physical constants
B_THEORY = 1 / (2 * np.pi)  # 0.15915


def main():
    print("="*60)
    print("Generate publication plot using ALPAKA results only")
    print("="*60)

    # Read ALPAKA data
    df_alpaka = pd.read_csv("data/alpaka_rar.csv")
    df_alpaka.columns = df_alpaka.columns.str.strip()
    z_alpaka = df_alpaka["z"].values
    B_alpaka = df_alpaka["B_corr"].values
    B_alpaka_err = df_alpaka["B_corr_err"].values

    print(f"\nALPAKA high-redshift sample results:")
    print(f"  Number of galaxies: {len(df_alpaka)}")
    print(f"  Redshift range: {np.min(z_alpaka):.3f} - {np.max(z_alpaka):.3f}")
    print(f"  Mean B(z): {np.mean(B_alpaka):.4f}")
    rel_dev = (np.mean(B_alpaka) - B_THEORY) / B_THEORY * 100
    print(f"  Relative deviation from theory {np.round(B_THEORY,4)}: {rel_dev:.1f}%")

    # Generate core publication plot
    print("\nGenerating publication plot...")
    plt.figure(figsize=(16, 8), dpi=150)

    plt.axhline(y=B_THEORY, color='red', linestyle='--', linewidth=2,
                label=r'Theoretical prediction $1/(2\pi)$')
    plt.errorbar(z_alpaka, B_alpaka, yerr=B_alpaka_err, fmt='o',
                 color='blue', markersize=8, capsize=5, label='High-z sample (ALPAKA)')

    plt.xlabel('Redshift z', fontsize=14)
    plt.ylabel('B(z)', fontsize=14)
    plt.xlim(-0.1, 2.4)
    plt.ylim(0, 0.4)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('Bz_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig('Bz_plot.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # Save ALPAKA results as LaTeX table
    df_alpaka[["name", "z", "g_obs", "g_bar", "B_corr", "B_corr_err"]].to_latex(
        'alpaka_result_table.tex',
        index=False,
        caption='ALPAKA high-redshift sample B(z) results',
        label='tab:alpaka',
        float_format=lambda x: f"{x:.3f}"
    )

    print("\n" + "="*60)
    print("Done! Publication plot and table saved.")
    print("="*60)


if __name__ == "__main__":
    main()
