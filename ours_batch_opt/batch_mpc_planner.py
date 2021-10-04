

import numpy as np
import jax.numpy as jnp
import bernstein_coeff_order10_arbitinterval
import scipy.io
from scipy.linalg import block_diag
import optim_elliptical_mpc
import wrapper_batch_mpc
import jax

from jax import jit, random
from jax.ops import index_update, index
from functools import partial
# from jax.config import config; config.update("jax_enable_x64", True)


import time
import matplotlib.pyplot as plt 
import scipy.io
from scipy.io import loadmat



class Traj_Opt_Elliptical():

	def __init__(self, num_goal, psi_des, t_fin, lx, a_obs_elliptical, maxiter_global, weight_smoothness, rho_ineq, rho_obs_elliptical, rho_w, rho_psi, weight_psi, rho_track):

		self.maxiter_elliptical = maxiter_global
		self.weight_smoothness = weight_smoothness
		self.t_fin = t_fin
		self.num = 100
		self.t = self.t_fin/self.num																			# dt
		self.tot_time = np.linspace(0.0, self.t_fin, self.num)
		tot_time_copy = self.tot_time.reshape(self.num,1)
		self.P, self.Pdot, self.Pddot = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)
		
		self.P_jax = jnp.asarray(self.P)
		self.Pdot_jax = jnp.asarray(self.Pdot)
		self.Pddot_jax = jnp.asarray(self.Pddot)		
		self.psi_des = psi_des

		self.nvar = np.shape(self.P)[1]

		##################### initial and final conditions
		self.lx = lx
		self.num_goal = num_goal
		
		smooth_mat = np.diff(self.Pddot, axis = 0)

		self.cost_smoothness = self.weight_smoothness*np.dot(self.Pddot.T, self.Pddot)
		

		########################################
		self.A_eq_prim = np.vstack(( self.P[0], self.Pdot[0], self.Pddot[0], self.P[-1], self.Pdot[-1], self.Pddot[-1]  ))

		self.A_eq = np.vstack(( self.P[0], self.Pdot[0], self.Pddot[0], self.P[-1] ))

		A_eq_psi_mat =  np.vstack(( self.P[0], self.Pdot[0], self.Pddot[0] ))
		
		self.A_eq_mat = np.vstack(( self.P[0], self.Pdot[0], self.Pddot[0], self.P[-1], self.Pdot[-1], self.Pddot[-1]  ))
		
		######################### Smoothness weights and proximal weights

		self.weight_psi = weight_psi
		self.rho_obs_elliptical = rho_obs_elliptical
		self.rho_w = rho_w
		self.rho_psi = rho_psi


		################################ Elliptical/Multi-circle Part

		self.a_obs_elliptical = a_obs_elliptical
		self.b_obs_elliptical = a_obs_elliptical
	
		self.A_eq_elliptical = jnp.hstack(( jnp.asarray(self.A_eq), jnp.zeros(( jnp.shape(self.A_eq)[0], self.nvar  ))    ))
		self.A_eq_psi = jnp.asarray(A_eq_psi_mat) 
		
		
		self.cost_smoothness_psi = self.weight_psi*jnp.dot(self.Pddot_jax.T, self.Pddot_jax)+0.1*self.weight_psi*jnp.dot(self.P_jax.T, self.P_jax)
		self.lincost_smoothness_psi = -0.1*self.weight_psi*jnp.dot(self.P_jax.T, self.psi_des*jnp.ones(( self.num_goal, self.num   )).T  ).T	
		
		self.num_circle = 4
		self.num_obs = 30

		self.A_w = jnp.hstack(( jnp.zeros(( self.num, self.nvar   )), self.P_jax        ))

		self.cost_smoothness_elliptical = jnp.asarray(block_diag(self.cost_smoothness, 0.0001*self.cost_smoothness))
	
		self.lamda_x_elliptical = jnp.zeros(( self.num_goal, 2*self.nvar))
		self.lamda_y_elliptical = jnp.zeros(( self.num_goal, 2*self.nvar))
		self.lamda_psi = jnp.zeros(( self.num_goal, self.nvar ))


		#################  Creating  obstacle matrices
		block_mat = jnp.tile(self.P_jax, (self.num_obs, 1)    )

		heading_block = jnp.vstack(( (self.lx/8)*block_mat,  (-self.lx/8)*block_mat, (3*self.lx/8)*block_mat, (-3*self.lx/8)*block_mat  ))
		x_block = jnp.vstack(( block_mat, block_mat, block_mat, block_mat   ))
		self.A_obs_elliptical = jnp.hstack(( x_block, heading_block ))

		########################## Velocity and acceleration bound parameters 

		self.a_max = 1.0
		self.v_max = 2.0
		#self.v_max = 2.0 #### for pedsim straight

		self.A_vel = jnp.hstack(( self.Pdot_jax, jnp.zeros(( self.num, self.nvar  ))  ))
		self.A_acc = jnp.hstack(( self.Pddot_jax, jnp.zeros(( self.num, self.nvar  ))  ))

		self.rho_ineq = rho_ineq

		self.d_v = self.v_max*jnp.ones((self.num_goal, self.num))
		self.d_a = self.a_max*jnp.ones((self.num_goal, self.num))

		self.alpha_v = jnp.zeros((self.num_goal, self.num))
		self.alpha_a = jnp.zeros((self.num_goal, self.num))

		################################################## Tracking Term

		self.A_track = jnp.hstack(( self.P_jax, jnp.zeros(( self.num, self.nvar  ))      ))

		self.alpha_track = jnp.zeros(( self.num_goal, self.num) ) 
		self.d_track = jnp.ones(( self.num_goal, self.num) ) 
		self.d_min_track = 3.0
		self.rho_track = rho_track

################### creating jit compiled versions of the function. This gets compiled on the first call
	def jit_compile(self):

		self.compute_x_elliptical_jit = jit(optim_elliptical_mpc.compute_x_elliptical, static_argnums=([0, 1, 2, 3, 4]))
		self.compute_psi_jit = jit(optim_elliptical_mpc.compute_psi, static_argnums=(0))
		self.compute_alpha_d_jit = jit(optim_elliptical_mpc.compute_alpha_d, static_argnums=(0, 1, 2, 3))
		self.bounds_comp_jit = jit(optim_elliptical_mpc.bounds_comp, static_argnums=(0, 1))
		self.compute_residuals_jit = jit(optim_elliptical_mpc.compute_residuals)
		self.initialize_guess_alpha_samples_jit = jit(wrapper_batch_mpc.initialize_guess_alpha_samples, static_argnums=(0,))
		self.obstacle_prediction_jit = jit(wrapper_batch_mpc.obstacle_prediction, static_argnums=(0,) )
		self.compute_boundary_vec_jit = jit(wrapper_batch_mpc.compute_boundary_vec, static_argnums=(0,) )
		self.jax_where_jit = jit(wrapper_batch_mpc.jax_where)




def compute_mpc(prob, mpc_iter, x_init, vx_init, ax_init, y_init, vy_init, ay_init, x_fin, y_fin, psi_init, psidot_init, psiddot_init, x_obs_track, y_obs_track, vx_obs_track, vy_obs_track, psi_noise, eps_k, vx_des, vy_des, x_des_traj_init, y_des_traj_init):

	prob.psi = prob.psi_des+psi_noise

	xdot_samples = jnp.zeros((prob.num_goal, prob.num))
	ydot_samples = jnp.zeros((prob.num_goal, prob.num))

	xddot_samples = jnp.zeros((prob.num_goal, prob.num))
	yddot_samples = jnp.zeros((prob.num_goal, prob.num))


	
	x_fin = x_des_traj_init+vx_des*prob.t_fin
	y_fin = y_des_traj_init+vy_des*prob.t_fin
	
	x_interp = jnp.linspace(x_des_traj_init, x_fin, prob.num)
	y_interp = jnp.linspace(y_des_traj_init, y_fin, prob.num)

	x_samples = x_interp+0.0*eps_k 
	y_samples = y_interp+eps_k 
	
	x_obs_init = x_obs_track[:, mpc_iter]
	y_obs_init = y_obs_track[:, mpc_iter]

	vx_obs = vx_obs_track[:, mpc_iter]
	vy_obs = vy_obs_track[:, mpc_iter]


	x_des_traj = x_des_traj_init+vx_des*prob.tot_time
	y_des_traj = y_des_traj_init+vy_des*prob.tot_time


	x_obs, y_obs = prob.obstacle_prediction_jit(prob.num_circle, x_obs_init, y_obs_init, vx_obs, vy_obs, prob.tot_time  )
	prob.alpha_obs_elliptical, prob.d_obs_elliptical, prob.alpha_v, prob.d_v, prob.alpha_a, prob.d_a = prob.initialize_guess_alpha_samples_jit(prob.num_goal, x_samples, y_samples, x_obs, y_obs, xdot_samples, ydot_samples, xddot_samples, yddot_samples, prob.v_max, prob.a_max, prob.A_obs_elliptical, prob.A_w, prob.a_obs_elliptical, prob.lamda_x_elliptical, prob.lamda_y_elliptical, prob.A_vel, prob.A_acc )

	prob.b_x_eq, prob.b_y_eq, prob.b_eq_psi = prob.compute_boundary_vec_jit(prob.num_goal, x_init, vx_init, ax_init, x_fin, y_init, vy_init, ay_init, y_fin, psi_init, psidot_init, psiddot_init)

	sol_x, sol_y, sol_psi, res_x_obs_vec, res_y_obs_vec, x_elliptical, y_elliptical, xddot_elliptical, yddot_elliptical, res_vx_vec, res_vy_vec, res_ax_vec, res_ay_vec, psi_elliptical, xdot_elliptical, ydot_elliptical, res_x_track_vec, res_y_track_vec, res_x_track_res, res_y_track_res, psiddot_elliptical = wrapper_batch_mpc.solve(prob, psi_init, psidot_init, psiddot_init, x_obs, y_obs, x_des_traj, y_des_traj)

	min_res_track = 0.01*(jnp.linalg.norm(xddot_elliptical, axis =1)+jnp.linalg.norm(yddot_elliptical, axis =1))+jnp.linalg.norm( jnp.hstack((  res_x_track_res, res_y_track_res       )), axis = 1          )+jnp.linalg.norm( jnp.hstack((  res_vx_vec, res_vy_vec, res_ax_vec, res_ay_vec       )), axis = 1          )+0.1*jnp.linalg.norm(psiddot_elliptical, axis = 1)+0.1*jnp.linalg.norm(psi_elliptical-prob.psi_des, axis = 1)

	min_res_obs_temp = jnp.linalg.norm( jnp.hstack((  res_x_obs_vec, res_y_obs_vec       )), axis = 1          )

	idx = prob.jax_where_jit(min_res_obs_temp)
	idx_obs = jnp.argmin(min_res_obs_temp)

	return np.asarray(sol_x), np.asarray(sol_y), np.asarray(sol_psi), np.asarray(idx), np.asarray(min_res_track), np.asarray(idx_obs), np.asarray(min_res_obs_temp)


