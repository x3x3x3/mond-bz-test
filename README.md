# mond-bz-test

# A test of the cosmological scaling of MOND's critical acceleration

This repository contains the code and data for the paper:
> Wenhao Xiong, 2026, MNRAS Letters, submitted

## Abstract
Modified Newtonian Dynamics (MOND) is built on a universal critical acceleration $a_0$, with a long-standing numerical coincidence between the local best-fit $a_0$ and $cH_0/(2\pi)$. We define a dimensionless observable $\mathcal{B}(z) \equiv a_{\rm tot}^2/(c H(z) a_{\rm N})$, which cancels structural systematics and directly probes $a_0(z)/(cH(z))$ in the weak-acceleration regime.

We test the prediction that $\mathcal{B}(z)$ is a universal constant $1/(2\pi) \approx 0.159$ with three independent datasets:
1. 175 nearby SPARC galaxies
2. 1207 radial acceleration points from the SPARC sample in the strict deep-MOND regime
3. 5 high-redshift regular rotating disk galaxies ($z=0.56-2.10$) from the ALMA ALPAKA survey

We find that current data are consistent with the hypothesis, with no statistically significant evidence for redshift evolution.

## Repository Structure
- `main.tex`: Main MNRAS Letter manuscript
- `supplementary.tex`: Supplementary material
- `references.bib`: BibTeX references
- `code/`: Python code for data analysis and plotting
- `image.png`: Figure 1 in the paper

## Quick Start
```bash
pip install numpy pandas scipy matplotlib
cd code
python mond_bz_analysis.py     
python plot_figure.py      
