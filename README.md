# Cosmic Scaling of Galaxy Dynamics

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
