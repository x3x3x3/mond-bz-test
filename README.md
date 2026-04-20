# Cosmic Scaling of Galaxy Dynamics

This repository contains the complete analysis code, processed data, and figure-generation scripts for the paper:

**"Local Galaxy Dynamics Normalized by the Cosmic Acceleration Scale"**  
*Wenhao Xiong*  
Submitted to MNRAS Letters (2026)

## Overview

We test the hypothesis that galaxy dynamics are globally normalized by the cosmic acceleration scale $a_z = cH_0/(2\pi)$. Using 3252 independent radial points from 161 SPARC galaxies, we compute dimensionless variables $x \equiv a_N/a_z$ and $y \equiv a_{\rm tot}/a_z$. The data collapse onto a tight empirical relation $y = F(x)$ with an intrinsic scatter of $\sim 0.12$ dex. A shuffle test ($p < 10^{-4}$) confirms the collapse is not an algebraic artifact. We also infer a dynamical Hubble constant $H_0^{\rm dyn}$ from each galaxy, finding median values consistent with Planck.

**Key results:**
- Median $x = 0.220 \pm 0.012$, with 75% of points in the weak-field regime ($x < 1$)
- Spearman $\rho = 0.872$ between $\log x$ and $\log y$
- Unfiltered $H_0^{\rm dyn} = 61.8 \pm 45.5$ km/s/Mpc (MAD)
- Filtered $H_0^{\rm dyn} = 61.0 \pm 36.6$ km/s/Mpc (MAD)


## Data Sources

- **SPARC**: [Spitzer Photometry and Accurate Rotation Curves](http://astroweb.cwru.edu/SPARC/) (Lelli et al. 2016, AJ, 152, 157). The required files are `table1.dat` and `table2.dat`. Place them in `data/SPARC/`.
- **HiZELS** (illustrative only): Kinematic data from Gillman et al. (2019, MNRAS, 486, 175). The files `hizels_kinematics.csv` and `hizels_properties.csv` are optional; if not present, the script will skip the HiZELS overlay and still run successfully.
- 

Install the dependencies via pip:


pip install numpy pandas matplotlib scipy

python code/final_scaling_analysis.py






## Outputs:

figure1_collapse_with_insets.pdf — main collapse figure

figure2_shuffle_test.pdf — shuffle test figure

supp_combined.pdf — supplementary 2x2 panel figure

sparc_scaling_data.csv — all processed data points

Console output with key statistics (medians, correlations, $H_0$ values)



## Repository Structure
├── code/final_scaling_analysis.py # Main analysis script (generates all figures and outputs)

├── sparc_scaling_data.csv # Processed data for all 3252 radial points

├── figure1_collapse_with_insets.pdf # Main figure 1: scaling collapse + insets

├── figure2_shuffle_test.pdf # Main figure 2: shuffle test

├── supp_combined.pdf # Supplementary figure (2x2 panel)

├── main.tex # LaTeX source for the manuscript

├── supplementary.tex # LaTeX source for supplementary material

├── references.bib # Bibliography

├── README.md # This file

└── data/

├── hizels_kinematics.csv # HiZELS kinematics (Gillman+2019) — optional

├── hizels_properties.csv # HiZELS properties (Gillman+2019) — optional

├── SPARC/

│     ├── table1.dat # SPARC galaxy sample (from Lelli+2016)

│     ├── table2.dat # SPARC mass models (from Lelli+2016)

