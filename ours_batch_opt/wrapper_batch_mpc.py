

import jax.numpy as jnp
from jax.ops import index_update, index
import jax

# import matplotlib.pyplot as plt 



def jax_where(p):

	return jnp.where(p<0.01, size = 1000)



############# function for linear-prediction of the obstacle
def obstacle_prediction(num_circle, x_obs_init, y_obs_init, vx_obs, vy_obs, tot_time  ):

	x_temp = x_obs_init+vx_obs*tot_time[:,jnp.newaxis]
	x_obs = x_temp.T 

	y_temp = y_obs_init+vy_obs*tot_time[:,jnp.newaxis]
	y_obs = y_temp.T

	x_obs_elliptical = jnp.tile(x_obs, (num_circle, 1)  )
	y_obs_elliptical = jnp.tile(y_obs, (num_circle, 1)  )


	return x_obs, y_obs



############ initializing alpha, d parameters from the paper
def initialize_guess_alpha_samples(num_goal, x_guess, y_guess, x_obs, y_obs, xdot_samples, ydot_samples, xddot_samples, yddot_samples, v_max, a_max, A_obs_elliptical, A_w, a_obs_elliptical, lamda_x_elliptical, lamda_y_elliptical, A_vel, A_acc ):

		
		wc_alpha_temp = (x_guess-x_obs[:,jnp.newaxis])
		ws_alpha_temp = (y_guess-y_obs[:,jnp.newaxis])

		wc_alpha = jnp.transpose((x_guess-x_obs[:,jnp.newaxis] ),[1,0,2]).reshape(num_goal, 100*30)
		ws_alpha = jnp.transpose((y_guess-y_obs[:,jnp.newaxis] ),[1,0,2]).reshape(num_goal, 100*30)

		alpha_obs = jnp.arctan2( ws_alpha, wc_alpha)

		rho_obs = 1.0
		a_obs = 1.0
		b_obs = 1.0

		c1_d = 1.0*rho_obs*(jnp.cos(alpha_obs)**2 +jnp.sin(alpha_obs)**2 )
		c2_d = 1.0*rho_obs*(wc_alpha*jnp.cos(alpha_obs) + ws_alpha*jnp.sin(alpha_obs)  )

		d_temp = c2_d/c1_d
		d_obs_temp = jnp.maximum(jnp.ones((num_goal,  100*30)), d_temp   )
		d_obs = d_obs_temp
		# self.d_obs = jnp.ones((self.num_goal,  self.num*self.num_obs))

		alpha_obs_elliptical = jnp.tile(alpha_obs, 4  )
		d_obs_elliptical = jnp.tile(d_obs, 4   )

		######velocity  bounds 
		rho_ineq = 1
		wc_alpha_vx = xdot_samples
		ws_alpha_vy = ydot_samples 
		alpha_v = jnp.arctan2( ws_alpha_vy, wc_alpha_vx) 

		c1_d_v = 1.0 * rho_ineq * (jnp.cos(alpha_v)**2 + jnp.sin(alpha_v)**2 )
		c2_d_v = 1.0 * rho_ineq * (wc_alpha_vx * jnp.cos(alpha_v) + ws_alpha_vy * jnp.sin(alpha_v)  )

		d_temp_v = c2_d_v/c1_d_v
		d_v = jnp.minimum(v_max *jnp.ones((num_goal, 100)), d_temp_v )

		###########acceleration bound 
		wc_alpha_ax = xddot_samples
		ws_alpha_ay = yddot_samples
		alpha_a = jnp.arctan2( ws_alpha_ay, wc_alpha_ax)
		   
		c1_d_a = 1.0 * rho_ineq * (jnp.cos(alpha_a)**2 + jnp.sin(alpha_a)**2 )
		c2_d_a = 1.0 * rho_ineq * (wc_alpha_ax * jnp.cos(alpha_a) + ws_alpha_ay * jnp.sin(alpha_a)  )

		d_temp_a = c2_d_a/c1_d_a
		d_a = jnp.minimum(a_max*jnp.ones((num_goal, 100)), d_temp_a  )

		#################################### Tracking bound
		wc_alpha_elliptical = jnp.tile(wc_alpha, 4)
		ws_alpha_elliptical = jnp.tile(ws_alpha, 4)

		b_obs_elliptical = a_obs_elliptical

		res_x_obs_vec = wc_alpha_elliptical-a_obs_elliptical*d_obs_elliptical*jnp.cos(alpha_obs_elliptical)
		res_y_obs_vec = ws_alpha_elliptical-b_obs_elliptical*d_obs_elliptical*jnp.sin(alpha_obs_elliptical)

		res_vx_vec = xdot_samples - d_v * jnp.cos(alpha_v)
		res_ax_vec = xddot_samples - d_a * jnp.cos(alpha_a)

		res_vy_vec = ydot_samples - d_v * jnp.sin(alpha_v)
		res_ay_vec = yddot_samples - d_a * jnp.sin(alpha_a)



		lamda_x_elliptical = lamda_x_elliptical-jnp.dot(A_obs_elliptical.T, res_x_obs_vec.T).T - jnp.dot( A_vel.T, res_vx_vec.T).T -jnp.dot( A_acc.T, res_ax_vec.T).T#- rho_track * jnp.dot( A_track.T, res_x_track_vec.T).T
		lamda_y_elliptical = lamda_y_elliptical-jnp.dot(A_obs_elliptical.T, res_y_obs_vec.T).T -  jnp.dot( A_vel.T, res_vy_vec.T).T -jnp.dot( A_acc.T, res_ay_vec.T).T#- rho_track * jnp.dot( A_track.T, res_y_track_vec.T).T




		
		return alpha_obs_elliptical, d_obs_elliptical, alpha_v, d_v, alpha_a, d_a





def solve(prob, psi_init, psidot_init, psiddot_init, x_obs, y_obs, x_des_traj, y_des_traj):

	res_psi_norm = []#jnp.zeros(prob_data.maxiter_elliptical)
	res_w_norm = []
	res_obs_norm = []
	res_bound_norm = []

	
	for i in range(0, prob.maxiter_elliptical):

			# start = time.time()

			wc, ws, x_elliptical, xdot_elliptical, y_elliptical, ydot_elliptical, xddot_elliptical, yddot_elliptical, sol_x, sol_y, x_des_traj_batch, y_des_traj_batch = prob.compute_x_elliptical_jit(prob.num_goal, prob.num, prob.num_obs, prob.num_circle, prob.nvar, prob.P_jax, prob.Pdot_jax, prob.Pddot_jax, prob.d_v, prob.d_a, prob.alpha_v, prob.alpha_a, prob.rho_ineq, prob.A_vel, prob.A_acc, prob.d_obs_elliptical, prob.alpha_obs_elliptical, prob.a_obs_elliptical, prob.b_obs_elliptical, prob.psi, prob.cost_smoothness_elliptical, prob.rho_obs_elliptical, prob.A_obs_elliptical, prob.rho_w, prob.A_w, prob.lamda_x_elliptical, prob.lamda_y_elliptical, prob.b_x_eq, prob.b_y_eq, prob.A_eq_elliptical, x_obs, y_obs, x_des_traj, y_des_traj, prob.rho_track, prob.A_track, prob.d_track, prob.alpha_track) 																								
			res_psi, prob.psi, prob.lamda_psi, sol_psi = prob.compute_psi_jit(prob.nvar, prob.P_jax, prob.Pdot_jax, wc, ws, prob.cost_smoothness_psi, prob.lamda_psi, prob.rho_psi, prob.b_eq_psi, prob.A_eq_psi, prob.lincost_smoothness_psi  )
			d_min, wc_alpha, ws_alpha, prob.d_obs_elliptical, prob.alpha_obs_elliptical  = prob.compute_alpha_d_jit( prob.num, prob.num_obs, prob.num_circle, prob.num_goal, prob.a_obs_elliptical, prob.b_obs_elliptical, x_elliptical, y_elliptical, prob.lx, prob.psi, x_obs, y_obs, prob.rho_obs_elliptical )	  

			prob.alpha_v, prob.d_v, prob.alpha_a, prob.d_a, res_vx_vec, res_ax_vec, res_vy_vec, res_ay_vec, prob.alpha_track, prob.d_track, res_x_track_vec, res_y_track_vec, res_x_track_res, res_y_track_res =  prob.bounds_comp_jit(prob.num_goal, prob.num, xdot_elliptical, ydot_elliptical, prob.rho_ineq, prob.v_max, xddot_elliptical, yddot_elliptical, prob.a_max, x_elliptical, y_elliptical, x_des_traj_batch, y_des_traj_batch, prob.rho_track, prob.d_min_track)
			res_w, res_obs, res_bound, prob.lamda_y_elliptical, prob.lamda_x_elliptical, res_x_obs_vec, res_y_obs_vec = prob.compute_residuals_jit(prob.rho_ineq, prob.A_vel, prob.A_acc, res_vx_vec, res_ax_vec, res_vy_vec, res_ay_vec, wc, ws, wc_alpha, ws_alpha, prob.a_obs_elliptical, prob.b_obs_elliptical, prob.alpha_obs_elliptical, prob.psi, prob.lamda_x_elliptical, prob.lamda_y_elliptical, prob.rho_obs_elliptical, prob.A_obs_elliptical, prob.rho_w, prob.A_w, prob.d_obs_elliptical, prob.rho_psi, prob.rho_track, prob.alpha_track, prob.d_track, res_x_track_vec, res_y_track_vec, prob.A_track   )

			psiddot_elliptical = jnp.dot(prob.P_jax, sol_psi.T).T


			
	return sol_x, sol_y, sol_psi, res_x_obs_vec, res_y_obs_vec, x_elliptical, y_elliptical, xddot_elliptical, yddot_elliptical, res_vx_vec, res_vy_vec, res_ax_vec, res_ay_vec, prob.psi, xdot_elliptical, ydot_elliptical, res_x_track_vec, res_y_track_vec, res_x_track_res, res_y_track_res, psiddot_elliptical





def compute_boundary_vec(num_goal, x_init, vx_init, ax_init, x_fin, y_init, vy_init, ay_init, y_fin, psi_init, psidot_init, psiddot_init):

	x_init_vec = x_init*jnp.ones((num_goal, 1))
	y_init_vec = y_init*jnp.ones((num_goal, 1)) 


	x_fin_vec = x_fin*jnp.ones((num_goal, 1))
	y_fin_vec = y_fin*jnp.ones((num_goal, 1)) 

	vx_init_vec = vx_init*jnp.ones((num_goal, 1))
	vy_init_vec = vy_init*jnp.ones((num_goal, 1))

	ax_init_vec = ax_init*jnp.ones((num_goal, 1))
	ay_init_vec = ay_init*jnp.ones((num_goal, 1))

	##############Heading angles conditions
	psi_init_vec =  psi_init*jnp.ones((num_goal, 1))
	
	psidot_init_vec = psidot_init*jnp.ones((num_goal, 1))
	
	psiddot_init_vec = psiddot_init*jnp.ones((num_goal, 1))
	
	b_x_eq = jnp.hstack(( x_init_vec, vx_init_vec, ax_init_vec, x_fin_vec  ))
	b_y_eq = jnp.hstack(( y_init_vec, vy_init_vec, ay_init_vec, y_fin_vec ))
	# b_eq_psi = jnp.hstack(( psi_init_vec, psidot_init_vec, psiddot_init_vec, psi_fin_vec, psidot_fin_vec, psiddot_fin_vec ))
	b_eq_psi = jnp.hstack(( psi_init_vec, psidot_init_vec, psiddot_init_vec ))

	return b_x_eq, b_y_eq, b_eq_psi



