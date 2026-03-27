# mond-bz-test

# A test of the cosmological scaling of MOND's critical acceleration

This repository contains the full reproducibility materials for the paper:
> **Wenhao Xiong**, 2026, *Monthly Notices of the Royal Astronomical Society Letters*, submitted.

---

## Abstract
Modified Newtonian Dynamics (MOND) is built on a universal critical acceleration $a_0$, with a long-standing numerical coincidence between the local best-fit $a_0$ and $cH_0/(2\pi)$. Previous tests of $a_0$'s cosmic evolution have been limited by baryonic modeling systematics and the narrow applicability of the deep-MOND approximation.

Here we define a dimensionless observable $\mathcal{B}(z) \equiv a_{\rm tot}^2/(c H(z) a_{\rm N})$, which cancels structural systematics and directly probes $a_0(z)/(cH(z))$ in the weak-acceleration regime. Under the hypothesis $a_0(z) = cH(z)/(2\pi)$, $\mathcal{B}(z)$ is predicted to be a universal constant $1/(2\pi) \approx 0.159$.

We test this prediction with three complementary, published datasets:
1.  **175 nearby SPARC galaxies** (Lelli et al. 2016)
2.  **1207 radial acceleration points** from the SPARC sample in the strict deep-MOND regime ($a_{\rm tot} < 0.1a_0$)
3.  **5 high-redshift regular rotating disk galaxies** ($z=0.561-2.103$) from the ALMA ALPAKA survey (Rizzo et al. 2023), selected as the 5 D-class galaxies with the highest kinematic quality

We find that current data are consistent with the hypothesis, with no statistically significant evidence for redshift evolution. Our findings serve as a preliminary consistency check for the proposed cosmological scaling of $a_0$.

---

## Repository Structure
├── README.md # This file

├── LICENSE # MIT License

├── main.tex # Main MNRAS Letter manuscript (as submitted)

├── supplementary.tex # Supplementary material (as submitted)

├── references.bib # BibTeX references

├── image.png # Figure 1 in the paper

└── code/

└── mond_bz_analysis.py # Core analysis and plotting functions

└── data/



## Quick Start
```bash
pip install numpy pandas scipy matplotlib
cd code
python mond_bz_analysis.py     
