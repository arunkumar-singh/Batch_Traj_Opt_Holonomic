
















import jax.numpy as jnp
import numpy as np 
from jax import jit



def compute_x_elliptical(num_goal, num, num_obs, num_circle, nvar, P, Pdot, Pddot, d_v, d_a, alpha_v, alpha_a, rho_ineq, A_vel, A_acc, d_obs_elliptical, alpha_obs_elliptical, a_obs_elliptical, b_obs_elliptical, psi, cost_smoothness_elliptical, rho_obs_elliptical, A_obs_elliptical, rho_w, A_w, lamda_x_elliptical, lamda_y_elliptical, b_x_eq, b_y_eq, A_eq_elliptical, x_obs, y_obs, x_des_traj, y_des_traj, rho_track, A_track, d_track, alpha_track):

	x_obs_elliptical = jnp.tile(x_obs, (num_circle, 1)  )
	y_obs_elliptical = jnp.tile(y_obs, (num_circle, 1)  )
		

	temp_x_obs = d_obs_elliptical*jnp.cos(alpha_obs_elliptical)*a_obs_elliptical
	b_obs_x_elliptical = x_obs_elliptical.reshape(100*30*4) + temp_x_obs

	temp_y_obs = d_obs_elliptical*jnp.sin(alpha_obs_elliptical)*b_obs_elliptical
	b_obs_y_elliptical = y_obs_elliptical.reshape(100*30*4) +temp_y_obs#reshape(num*num_obs*num_circle)

	b_wc = jnp.cos(psi)
	b_ws = jnp.sin(psi)

	b_vx_ineq = d_v * jnp.cos(alpha_v)
	b_ax_ineq = d_a * jnp.cos(alpha_a)

	b_vy_ineq = d_v * jnp.sin(alpha_v)
	b_ay_ineq = d_a * jnp.sin(alpha_a)

	x_des_traj_batch = jnp.tile( x_des_traj, (num_goal, 1))
	y_des_traj_batch = jnp.tile( y_des_traj, (num_goal, 1))
	
	b_x_track = x_des_traj_batch+d_track*jnp.cos(alpha_track)
	b_y_track = y_des_traj_batch+d_track*jnp.sin(alpha_track)

	# b_x_track = x_des_traj_batch
	# b_y_track = y_des_traj_batch
		

	lincost_x = -lamda_x_elliptical - rho_obs_elliptical * jnp.dot(A_obs_elliptical.T, b_obs_x_elliptical.T).T - rho_w * jnp.dot(A_w.T, b_wc.T).T - rho_ineq * jnp.dot(A_vel.T, b_vx_ineq.T).T - rho_ineq * jnp.dot(A_acc.T, b_ax_ineq.T).T-rho_track*jnp.dot(A_track.T, b_x_track.T).T
	lincost_y = -lamda_y_elliptical - rho_obs_elliptical * jnp.dot(A_obs_elliptical.T, b_obs_y_elliptical.T).T - rho_w * jnp.dot(A_w.T, b_ws.T).T - rho_ineq * jnp.dot(A_vel.T, b_vy_ineq.T).T - rho_ineq * jnp.dot(A_acc.T, b_ay_ineq.T).T-rho_track*jnp.dot(A_track.T, b_y_track.T).T

	# lincost_x = -lamda_x_elliptical - rho_obs_elliptical * jnp.dot(A_obs_elliptical.T, b_obs_x_elliptical.T).T - rho_w * jnp.dot(A_w.T, b_wc.T).T - rho_ineq * jnp.dot(A_vel.T, b_vx_ineq.T).T - rho_ineq * jnp.dot(A_acc.T, b_ax_ineq.T).T-rho_track*jnp.dot(A_track.T, x_des_traj_batch.T).T
	# lincost_y = -lamda_y_elliptical - rho_obs_elliptical * jnp.dot(A_obs_elliptical.T, b_obs_y_elliptical.T).T - rho_w * jnp.dot(A_w.T, b_ws.T).T - rho_ineq * jnp.dot(A_vel.T, b_vy_ineq.T).T - rho_ineq * jnp.dot(A_acc.T, b_ay_ineq.T).T-rho_track*jnp.dot(A_track.T, y_des_traj_batch.T).T


	cost = cost_smoothness_elliptical + rho_obs_elliptical * jnp.dot(A_obs_elliptical.T, A_obs_elliptical) + rho_w * jnp.dot(A_w.T, A_w) + rho_ineq * jnp.dot(A_vel.T, A_vel) + rho_ineq * jnp.dot(A_acc.T, A_acc)+rho_track*jnp.dot(A_track.T, A_track)
	cost_mat = jnp.vstack(( jnp.hstack(( cost, A_eq_elliptical.T )), jnp.hstack((A_eq_elliptical, jnp.zeros(( 4, 4 )) )) ))
	cost_mat_inv = jnp.linalg.inv(cost_mat)
	
	sol_x_temp = jnp.dot(cost_mat_inv, jnp.hstack(( -lincost_x, b_x_eq )).T ).T
	sol_x = sol_x_temp[:, 0:nvar]

	x_elliptical = jnp.dot(P, sol_x.T).T
	xdot_elliptical = jnp.dot(Pdot, sol_x.T).T
	xddot_elliptical = jnp.dot(Pddot, sol_x.T).T

	sol_y_temp = jnp.dot(cost_mat_inv, jnp.hstack(( -lincost_y, b_y_eq )).T ).T
	sol_y = sol_y_temp[:, 0:nvar]

	y_elliptical = jnp.dot(P, sol_y.T).T
	ydot_elliptical = jnp.dot(Pdot, sol_y.T).T
	yddot_elliptical = jnp.dot(Pddot, sol_y.T).T

	sol_x_temp1 = sol_x_temp[:, nvar:2*nvar]
	sol_y_temp1 = sol_y_temp[:, nvar:2*nvar]

	wc = jnp.dot(P, sol_x_temp1.T ).T
	ws = jnp.dot(P, sol_y_temp1.T ).T


	return wc, ws, x_elliptical, xdot_elliptical, y_elliptical, ydot_elliptical, xddot_elliptical, yddot_elliptical, sol_x, sol_y, x_des_traj_batch, y_des_traj_batch

def compute_psi(nvar, P, Pdot, wc, ws,  cost_smoothness_psi, lamda_psi, rho_psi, b_eq_psi, A_eq_psi, lincost_smoothness_psi   ):

    A_psi = P 
    b_psi = jnp.arctan2(ws, wc  )

    lincost = -lamda_psi-rho_psi*jnp.dot(A_psi.T, b_psi.T).T+lincost_smoothness_psi

    cost = cost_smoothness_psi+rho_psi*jnp.dot(P.T, P)
    cost_mat = jnp.vstack((  jnp.hstack(( cost, A_eq_psi.T )), jnp.hstack(( A_eq_psi, jnp.zeros(( 3,3 )) )) ))
    cost_mat_inv = jnp.linalg.inv(cost_mat)

    sol = jnp.dot(cost_mat_inv, jnp.hstack(( -lincost, b_eq_psi )).T ).T

    sol_psi = sol[:, 0:nvar]

    psi = jnp.dot(P, sol_psi.T).T

    lamda_psi_old = lamda_psi

    res_psi = jnp.dot(A_psi, sol_psi.T).T-b_psi
    lamda_psi = lamda_psi-rho_psi*jnp.dot(A_psi.T, res_psi.T).T

    # lamda_psi = lamda_psi+0.9*(lamda_psi-lamda_psi_old)


    return jnp.linalg.norm(res_psi), psi, lamda_psi, sol_psi

def compute_alpha_d(num, num_obs, num_circle, num_goal, a_obs_elliptical, b_obs_elliptical, x_elliptical, y_elliptical, lx, psi, x_obs, y_obs, rho_obs_elliptical   ):

    x_circ_1 = x_elliptical+(lx/8.0)*jnp.cos(psi)
    x_circ_2 = x_elliptical-(lx/8.0)*jnp.cos(psi)
    x_circ_3 = x_elliptical+(3*lx/8.0)*jnp.cos(psi)
    x_circ_4 = x_elliptical+(-3*lx/8.0)*jnp.cos(psi)
    
    y_circ_1 = y_elliptical+(lx/8.0)*jnp.sin(psi)
    y_circ_2 = y_elliptical-(lx/8.0)*jnp.sin(psi)
    y_circ_3 = y_elliptical+(3*lx/8.0)*jnp.sin(psi)
    y_circ_4 = y_elliptical+(-3*lx/8.0)*jnp.sin(psi)

    wc_alpha_circ_1 = jnp.transpose((x_circ_1-x_obs[:,jnp.newaxis] ),[1,0,2]).reshape(num_goal, 100*30)
    wc_alpha_circ_2 = jnp.transpose((x_circ_2-x_obs[:,jnp.newaxis] ),[1,0,2]).reshape(num_goal, 100*30)#.transpose(1, 0, 2)).reshape(num_goal, num*num_obs)
    wc_alpha_circ_3 = jnp.transpose((x_circ_3-x_obs[:,jnp.newaxis] ),[1,0,2]).reshape(num_goal, 100*30)#.transpose(1, 0, 2)).reshape(num_goal, num*num_obs)
    wc_alpha_circ_4 = jnp.transpose((x_circ_4-x_obs[:,jnp.newaxis] ),[1,0,2]).reshape(num_goal, 100*30)#.transpose(1, 0, 2)).reshape(num_goal, num*num_obs)
    
    ws_alpha_circ_1 = jnp.transpose((y_circ_1-y_obs[:,jnp.newaxis] ),[1,0,2]).reshape(num_goal, 100*30)#.transpose(1, 0, 2)).reshape(num_goal, num*num_obs)
    ws_alpha_circ_2 = jnp.transpose((y_circ_2-y_obs[:,jnp.newaxis] ),[1,0,2]).reshape(num_goal, 100*30)#.transpose(1, 0, 2)).reshape(num_goal, num*num_obs)
    ws_alpha_circ_3 = jnp.transpose((y_circ_3-y_obs[:,jnp.newaxis] ),[1,0,2]).reshape(num_goal, 100*30)#.transpose(1, 0, 2)).reshape(num_goal, num*num_obs)
    ws_alpha_circ_4 = jnp.transpose((y_circ_4-y_obs[:,jnp.newaxis] ),[1,0,2]).reshape(num_goal, 100*30)#.transpose(1, 0, 2)).reshape(num_goal, num*num_obs)

    wc_alpha = jnp.hstack(( wc_alpha_circ_1, wc_alpha_circ_2, wc_alpha_circ_3, wc_alpha_circ_4         ))
    ws_alpha = jnp.hstack(( ws_alpha_circ_1, ws_alpha_circ_2, ws_alpha_circ_3, ws_alpha_circ_4         ))
         
    alpha_obs_elliptical = jnp.arctan2( ws_alpha*a_obs_elliptical, wc_alpha*b_obs_elliptical   )

    c1_d = 1.0*rho_obs_elliptical*(a_obs_elliptical**2*jnp.cos(alpha_obs_elliptical)**2 + b_obs_elliptical**2*jnp.sin(alpha_obs_elliptical)**2 )
    c2_d = 1.0*rho_obs_elliptical*(a_obs_elliptical*wc_alpha*jnp.cos(alpha_obs_elliptical) + b_obs_elliptical*ws_alpha*jnp.sin(alpha_obs_elliptical)  )

    d_temp = c2_d/c1_d

    d_obs_elliptical = jnp.maximum(jnp.ones((num_goal,  100*30*4)), d_temp   )
    d_min = jnp.amin(d_temp)

    return d_min, wc_alpha, ws_alpha, d_obs_elliptical, alpha_obs_elliptical


def bounds_comp(num_goal, num, xdot_elliptical, ydot_elliptical, rho_ineq, v_max, xddot_elliptical, yddot_elliptical, a_max, x_elliptical, y_elliptical, x_des_traj_batch, y_des_traj_batch, rho_track, d_min_track):

	######velocity  bounds 
	wc_alpha_vx = xdot_elliptical 
	ws_alpha_vy = ydot_elliptical 
	alpha_v = jnp.arctan2( ws_alpha_vy, wc_alpha_vx) 

	c1_d_v = 1.0 * rho_ineq * (jnp.cos(alpha_v)**2 + jnp.sin(alpha_v)**2 )
	c2_d_v = 1.0 * rho_ineq * (wc_alpha_vx * jnp.cos(alpha_v) + ws_alpha_vy * jnp.sin(alpha_v)  )

	d_temp_v = c2_d_v/c1_d_v
	d_v = jnp.minimum(v_max *jnp.ones((num_goal, num)), d_temp_v )

	###########acceleration bound 
	wc_alpha_ax = xddot_elliptical
	ws_alpha_ay = yddot_elliptical
	alpha_a = jnp.arctan2( ws_alpha_ay, wc_alpha_ax)
	   
	c1_d_a = 1.0 * rho_ineq * (jnp.cos(alpha_a)**2 + jnp.sin(alpha_a)**2 )
	c2_d_a = 1.0 * rho_ineq * (wc_alpha_ax * jnp.cos(alpha_a) + ws_alpha_ay * jnp.sin(alpha_a)  )

	d_temp_a = c2_d_a/c1_d_a
	d_a = jnp.minimum(a_max*jnp.ones((num_goal, num)), d_temp_a  )

	#################################### Tracking bound

	wc_alpha_track = (x_elliptical-x_des_traj_batch)
	ws_alpha_track = (y_elliptical-y_des_traj_batch)
	alpha_track = jnp.arctan2( ws_alpha_track, wc_alpha_track )

	c1_d_track = 1.0*rho_track*(jnp.cos(alpha_track)**2 + jnp.sin(alpha_track)**2 )
	c2_d_track = 1.0*rho_track*(wc_alpha_track*jnp.cos(alpha_track) + ws_alpha_track*jnp.sin(alpha_track)  )

	d_temp_track = c2_d_track/c1_d_track
	d_track = jnp.minimum( d_temp_track, d_min_track*jnp.ones((num_goal, num )) )


	##################### residuals
	res_vx_vec = xdot_elliptical - d_v * jnp.cos(alpha_v)
	res_ax_vec = xddot_elliptical - d_a * jnp.cos(alpha_a)

	res_vy_vec = ydot_elliptical - d_v * jnp.sin(alpha_v)
	res_ay_vec = yddot_elliptical - d_a * jnp.sin(alpha_a)

	res_x_track_vec = wc_alpha_track - d_track * jnp.cos(alpha_track)
	res_y_track_vec = ws_alpha_track - d_track * jnp.sin(alpha_track)

	res_x_track_res = wc_alpha_track 
	res_y_track_res = ws_alpha_track 

	

	return alpha_v, d_v, alpha_a, d_a, res_vx_vec, res_ax_vec, res_vy_vec, res_ay_vec, alpha_track, d_track, res_x_track_vec, res_y_track_vec, res_x_track_res, res_y_track_res

def compute_residuals(rho_ineq, A_vel, A_acc, res_vx_vec, res_ax_vec, res_vy_vec, res_ay_vec, wc, ws, wc_alpha, ws_alpha, a_obs_elliptical, b_obs_elliptical, alpha_obs_elliptical, psi, lamda_x_elliptical, lamda_y_elliptical, rho_obs_elliptical, A_obs_elliptical, rho_w, A_w, d_obs_elliptical, rho_psi, rho_track, alpha_track, d_track, res_x_track_vec, res_y_track_vec, A_track):

	res_x_obs_vec = wc_alpha-a_obs_elliptical*d_obs_elliptical*jnp.cos(alpha_obs_elliptical)
	res_y_obs_vec = ws_alpha-b_obs_elliptical*d_obs_elliptical*jnp.sin(alpha_obs_elliptical)

	res_wc = wc-jnp.cos(psi)
	res_ws = ws-jnp.sin(psi)
	    
	# lamda_x_elliptical = lamda_x_elliptical-rho_obs_elliptical*jnp.dot(A_obs_elliptical.T, res_x_obs_vec.T).T-rho_w*jnp.dot(A_w.T, res_wc.T).T - rho_ineq * jnp.dot( A_vel.T, res_vx_vec.T).T - rho_ineq * jnp.dot( A_acc.T, res_ax_vec.T).T
	# lamda_y_elliptical = lamda_y_elliptical-rho_obs_elliptical*jnp.dot(A_obs_elliptical.T, res_y_obs_vec.T).T-rho_w*jnp.dot(A_w.T, res_ws.T).T - rho_ineq * jnp.dot( A_vel.T, res_vy_vec.T).T - rho_ineq * jnp.dot( A_acc.T, res_ay_vec.T).T

	lamda_x_elliptical = lamda_x_elliptical-rho_obs_elliptical*jnp.dot(A_obs_elliptical.T, res_x_obs_vec.T).T-rho_w*jnp.dot(A_w.T, res_wc.T).T - rho_ineq * jnp.dot( A_vel.T, res_vx_vec.T).T - rho_ineq * jnp.dot( A_acc.T, res_ax_vec.T).T- rho_track * jnp.dot( A_track.T, res_x_track_vec.T).T
	lamda_y_elliptical = lamda_y_elliptical-rho_obs_elliptical*jnp.dot(A_obs_elliptical.T, res_y_obs_vec.T).T-rho_w*jnp.dot(A_w.T, res_ws.T).T - rho_ineq * jnp.dot( A_vel.T, res_vy_vec.T).T - rho_ineq * jnp.dot( A_acc.T, res_ay_vec.T).T- rho_track * jnp.dot( A_track.T, res_y_track_vec.T).T


	res_obs = jnp.linalg.norm(jnp.hstack(( res_x_obs_vec, res_y_obs_vec   ))    )
	res_w = jnp.linalg.norm( jnp.hstack(( res_wc, res_ws     ))      )
	res_bound = jnp.linalg.norm(jnp.hstack(( res_vx_vec, res_ax_vec, res_vy_vec, res_ay_vec )) )

	
	    
	return res_w, res_obs, res_bound, lamda_y_elliptical, lamda_x_elliptical, res_x_obs_vec, res_y_obs_vec







    





    

        