# Conformable Fractional Physics-Informed Neural Networks (cfPINNs)

This repository contains the JAX implementation for my Master's Thesis, "Conformable Fractional Physics-Informed Neural Networks." It provides the source code for the cfPINN framework, as well as the scripts used to generate the results for all numerical experiments presented in the thesis.

## Abstract

Physics-Informed Neural Networks (PINNs) are a powerful framework for solving integer-order differential equations. However, their application to Fractional Differential Equations (FDEs) is challenged by the non-local nature of classical fractional operators (e.g., Caputo), which is incompatible with Automatic Differentiation (AD). The state-of-the-art solution, the Fractional PINN (fPINN), relies on numerical discretization for fractional terms, which introduces discretization error and faces scalability limitations.

This work proposes a novel, fully differentiable framework: the Conformable Fractional PINN (cfPINN). We leverage the local, AD-native improved conformable derivative to construct a surrogate FDE model that is solved end-to-end with a PINN. This approach eliminates the discretization error of fPINNs in exchange for an inherent surrogate modeling error. We demonstrate that for forward problems with sparse data, the cfPINN can learn an effective fractional order ($\hat{\alpha}$) that allows the local model to accurately approximate the non-local dynamics.

## Repository Structure

The repository is organized into a core framework and experiment scripts:

### Core Framework
*   `pinn_framework.py`: Contains the basic building blocks for the PINN, including the MLP architecture and the training step logic.
*   `dynamic_caputo_full.py`: The generalized, JIT-compatible implementation of the improved conformable Caputo-type derivative for any arbitrary order $\alpha > 0$.
*   `frac_laplacian_autodiff.py`: The high-performance implementation of the multi-dimensional fractional Laplacian surrogate, based on the directional conformable derivative.

### Numerical Experiments
The following scripts reproduce the results presented in the thesis:
*   `1D_IVP_fixed_alpha.py`: Solves the 1D Fractional IVP (forward problem) with a fixed fractional order to demonstrate the surrogate modeling error.
*   `1D_IVP_learnable_alpha.py`: Solves the 1D Fractional IVP (forward problem) by learning an effective fractional order from sparse data.
*   `1D_Poisson.py`: Solves the 1D Fractional Poisson problem by learning an effective fractional order.
*   `solve2DPoisson.py`: Solves the 2D time-dependent fractional advection-diffusion problem by learning an effective spatial order.

## Requirements

The code is built using JAX and Flax. 
To install all requirements run:
```bash
pip install -r requirements.txt
```

