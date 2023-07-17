# A continuous attractor network with multiple dynamic memories

![alt text](imgs/didascalia.png?raw=true)

This folder contains code for the numerical simulation and analytical solution of continuous attractor neural network with a dynamic shift mechanism, as presented in this paper: https://doi.org/10.7554/eLife.69499.

## Structure

- `Numerical-Simulations` contains code for the simulation of a single network dynamics. The dynamics can be implemented with different interaction kernels, hyperparameters and manifold dimensions. The code for the numerical calculation of the storage capacity of 1D and 2D dynamic continuous attractors can also be found here.  

- `Mean-Field` contains code for the numerical solution of the equations dervied analytically in the mean field approximation: the equations for the activity profile in the case of a single map, multiple maps, and the analytical calculation for the storage capacity of an asymmetric, highly diluted network.

## Usage

The package is based on base python and numpy. Download/Clone the repository and browse the code. Please refer to the paper for the theoretical details of the model.
