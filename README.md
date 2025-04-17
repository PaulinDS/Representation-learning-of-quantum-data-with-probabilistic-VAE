![Image](spinVAE_schema_archi2.png)


# Exploring-Latent-Representation-of-Quantum-Phase-Space-with-Variational-Auto-Encode
code used for the numerical simulation of the paper




## Structure

In [config.py](https://github.com/PaulinDS/Learning-Schmidt-decompositions-with-NN-and-QC/blob/main/config.py), there are the definitions of the Hamiltonian and the ARNN.

In [forging_functions.py](https://github.com/PaulinDS/Learning-Schmidt-decompositions-with-NN-and-QC/blob/main/forging_functions.py), there are functions used for the entanglement forging.

In [generative_algo_functions.py](https://github.com/PaulinDS/Learning-Schmidt-decompositions-with-NN-and-QC/blob/main/generative_algo_functions.py), there are functions used for the generative algorithm.

In [generative_algo_non_perm_sym.py](https://github.com/PaulinDS/Learning-Schmidt-decompositions-with-NN-and-QC/blob/main/generative_algo_non_perm_sym.py), there is an example of how to use the different functions to run the algorithm generating the set the bitstrings on a non permutation symetric system.

In [schrodinger_forging_VQE.py](https://github.com/PaulinDS/Learning-Schmidt-decompositions-with-NN-and-QC/blob/main/schrodinger_forging_VQE.py), there is an example of how to use the different functions to do the Schrodinger VQE. This part can be significantly accelerated with the use of GPUs.


## Package Versions

`flax 0.10.4`
`jax 0.4.38`
`jax_lib 0.4.38`
`optax 0.2.4`
`netket 3.15.2`




