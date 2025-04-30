# Exploring-Latent-Representation-of-Quantum-Phase-Space-with-Variational-Auto-Encode

![Image](spinVAE_schema_archi2.png)


Code used for the numerical simulation of the paper: ...




## Structure

In [modified_nk_ARNN.py](https://github.com/PaulinDS/Exploring-Latent-Representation-of-Quantum-Phase-Space-with-Variational-Auto-Encode/blob/main/modified_nk_ARNN.py), there is the implementation of the dense ARNN taken from netket and modified to be able to take additional inputs, the latent vectors. This implementation is used for the cpVAE on the spin models.

The code is presented as 3 notebooks, one for each quantum model.

In  [VAE_spins_NNNTFIM.ipynb](https://github.com/PaulinDS/Exploring-Latent-Representation-of-Quantum-Phase-Space-with-Variational-Auto-Encode/blob/main/VAE_spins_NNNTFIM.ipynb) and in  [VAE_spins_LRTFIM.ipynb](https://github.com/PaulinDS/Exploring-Latent-Representation-of-Quantum-Phase-Space-with-Variational-Auto-Encode/blob/main/VAE_spins_LRTFIM.ipynb), there is the entire code used for the simulations on the NNN-TFIM and the LR-TFIM, respectively. This includes the creation of the dataset with an exact diagonalization, the definition of the dVAE and the cpVAE, losses, training...

In  [VAE_spins_Rydberg.ipynb](https://github.com/PaulinDS/Exploring-Latent-Representation-of-Quantum-Phase-Space-with-Variational-Auto-Encode/blob/main/VAE_spins_Rydberg.ipynb) there is the entire code used for the simulations on the Rydberg atoms. This includes loading the experimental data the implementation of the cpVAE with transformer architecture, losses, training...


## Package Versions

`flax 0.10.4`
`jax 0.4.38`
`jax_lib 0.4.38`
`optax 0.2.4`
`netket 3.15.2`




