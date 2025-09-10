# Conformable Fractional Physics-Informed Neural Networks (cfPINNs)

This repository contains the JAX implementation for my Master's Thesis, "Conformable Fractional Physics-Informed Neural Networks." It provides the source code for the cfPINN framework, as well as the scripts used to generate the results for all numerical experiments presented in the thesis.

## Abstract
This thesis proposes a novel, fully differentiable framework: the Conformable Fractional PINN (cfPINN). 
We leverage the recent improved conformable derivative definition,
which is local and AD-compatible, to construct a surrogate FDE model that is solved entirely within the PINN architecture. This approach presents a fundamental trade-off: it eliminates the discretization error of fPINNs in exchange for an inherent surrogate modeling error.

We implement the cfPINN framework in JAX and validate it on benchmark FDEs, 
including 1D initial value problems, 1D fractional Poisson problems, 
and a 2D time-dependent advection-diffusion equation. 
We demonstrate that for forward problems where sparse data from the true non-local solution is available, 
the cfPINN can learn an effective fractional order $\hat{\alpha}$ that allows the local model to accurately 
approximate the non-local dynamics.
We conclude that the cfPINN is a viable alternative to fPINNs, offering a different set of trade-offs that may be advantageous, particularly for problems where high-dimensional scalability is critical.

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

## How to Run an Experiment

To run an experiment, simply execute the desired Python script. For example, to run the 1D IVP experiment with a learnable alpha:

```bash
python 1D_IVP_learnable_alpha.py
```

