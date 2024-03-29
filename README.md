# Batch_Traj_Opt_Holonomic
Repository associated with our ACC 2023 submission "GPU Accelerated Batch Trajectory Optimization for Autonomous Navigation". Preprint link will appear by soon.



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

Some Nuances of Jax

Jax uses the Just-in-Time Compilation (JIT). Thus, the first iteration of MPC is slower because Jax complies the code in the first call. Subsequently, the MPC runs at real-time speed of 0.04s on RTX 3080 laptop. 


How to Run the files
1. Navigate to ours_batch_opt folder to run the proposed batch trajectory optimization
2. A minimal mpc code is presented in main_batch_opt. It runs the MPC for a choosen obstacle data. 
3. The obstacle configuration can be changed by changing the respective files, as mentioned in main_batch_opt.py
4. The main_batch_opt_runall.py runs all the configurations for particular benchmark and stores the results in the form of mat file that can be analyzed latter for computing the success-rate, tracking efficiency, etc.



