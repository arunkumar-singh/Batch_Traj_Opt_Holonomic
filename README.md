# Batch_Traj_Opt_Holonomic
Repository associated with our ICRA 2022 submission. Codes and Preprints will appear by 30th Sept 2021
Arxiv Pre-Print https://arxiv.org/abs/2109.13030


Contacts: Arun Kumar Singh (aks1812@gmail.com), Fatemeh Rastgar (fatemeh@ut.ee)

Requirements:



1. Numpy.

2. Scipy.

3. Jax-Numpy (https://github.com/google/jax).

4. The code has been tested with CUDA version 11.1 and 11.2 on RTX-2080 (8GB) Desktop and RTX (3080) i7-8750 laptop computer with 32 GB RAM. 

Instructions to use the Code

The repository has two sets of codes

1. An implementation of Model Predictive Control built on top of our multi-convex batch optimizer

2. An implementation of Cross-Entropy Method for benchmarking

Both the implementation reads obstacle data from a mat file. The instaneous positons and velocities of the obstacle are read from the mat file and then the MPC constructs a linear approximation of the obstacle trajectories. For static obstacles, the velocities are zero and all entries in the trajectory of a particular obstacle are same.