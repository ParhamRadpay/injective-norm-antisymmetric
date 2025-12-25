# Antisymmetric Injective Norm Estimation

This repository contains Python code to generate random antisymmetric Gaussian tensors and numerically estimate their injective norm using gradient ascent.

For background and theoretical context, see (and references therein): arXiv:2510.25474

The script injective_norm_descent.py generates random antisymmetric tensors of order p and dimension d, computes an approximation of the injective norm, and saves results to JSON.

The notebook Plotter.ipynb provides tools to load multiple JSON result files, group data by dimension, plot mean ± standard deviation of the injective norm and apply different normalization schemes.

**Notes**:
Works best for tensor order p = 3 or 4\\
Optimization yields a numerical approximation\\
Random seeds influence results slightly

## Requirements
- Python 3.9+
- PyTorch
- NumPy
- tqdm
- Matplotlib (for plotting / notebooks)
