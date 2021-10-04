








import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt 
import time
import batch_mpc_planner

from scipy.io import loadmat

import scipy
import seaborn as sns
sns.set_theme(style="darkgrid")
import bernstein_coeff_order10_arbitinterval


############ function to do the collision check after the MPC simulation

def compute_min_dist(x_robot, y_robot, x_obs_track, y_obs_track, psi_robot, lx):

		##################################### computing minimum distance collision check

		dist_robot = np.zeros(np.shape(x_robot)[0])

		for i in range(0, np.shape(x_robot)[0]):

			x_circ_1 = x_robot[i]+(lx/8.0)*np.cos(psi_robot[i])
			x_circ_2 = x_robot[i]-(lx/8.0)*np.cos(psi_robot[i])
			x_circ_3 = x_robot[i]+(3*lx/8.0)*np.cos(psi_robot[i])
			x_circ_4 = x_robot[i]+(-3*lx/8.0)*np.cos(psi_robot[i])

			y_circ_1 = y_robot[i]+(lx/8.0)*np.sin(psi_robot[i])
			y_circ_2 = y_robot[i]-(lx/8.0)*np.sin(psi_robot[i])
			y_circ_3 = y_robot[i]+(3*lx/8.0)*np.sin(psi_robot[i])
			y_circ_4 = y_robot[i]+(-3*lx/8.0)*np.sin(psi_robot[i])

			wc_alpha_circ_1 = x_circ_1-x_obs_track[:,i] 
			wc_alpha_circ_2 = x_circ_2-x_obs_track[:,i] 
			wc_alpha_circ_3 = x_circ_3-x_obs_track[:,i] 
			wc_alpha_circ_4 = x_circ_4-x_obs_track[:,i] 


			ws_alpha_circ_1 = y_circ_1-y_obs_track[:,i]
			ws_alpha_circ_2 = y_circ_2-y_obs_track[:,i] 
			ws_alpha_circ_3 = y_circ_3-y_obs_track[:,i] 
			ws_alpha_circ_4 = y_circ_4-y_obs_track[:,i]

			dist_1 = np.sqrt(wc_alpha_circ_1**2+ws_alpha_circ_1**2)
			dist_2 = np.sqrt(wc_alpha_circ_2**2+ws_alpha_circ_2**2)
			dist_3 = np.sqrt(wc_alpha_circ_3**2+ws_alpha_circ_3**2)
			dist_4 = np.sqrt(wc_alpha_circ_4**2+ws_alpha_circ_4**2)

			dist_robot[i] =  np.min(  np. hstack(( dist_1, dist_2, dist_3, dist_4     ))  ) 
		# print(dist_robot[i])

		return dist_robot










num_goal = 1000 ####### batch size
# k = 16 ####### the obstacle configuration number k \in [0, 19]

for k in range(0, 20):
	############ For static obstacles, the following two files should be used

	# obs_file_name = 'obstacle_data_3/'+'obs_data_static_'+str(k+1)+'.mat'	

	obs_file_name = 'obstacle_data_4/'+'obs_data_static_'+str(k+1)+'.mat'	



	###################### The dynamic obstacles are in the following two files

	# obs_file_name = 'obstacle_data_1/'+'obs_data_dynamic_'+str(k+1)+'.mat'	

	# obs_file_name = 'obstacle_data_2/'+'obs_data_dynamic_'+str(k+1)+'.mat'	


	data = loadmat(obs_file_name)

	################################## State initialization

	x_init =  -3.0
	vx_init = 0.0
	ax_init = 0.0

	y_init =  0.0

	vy_init = 0.0
	ay_init = 0.0

	x_fin = 35.0
	vx_fin = 0.0
	ax_fin = 0.0

	y_fin = 0.0

	vy_fin = 0.0
	ay_fin = 0.0

	psi_init = 0*3.14/181.0
	psidot_init = 0.0
	psiddot_init = 0.0

	psi_fin = 0.0
	psidot_fin = 0.0
	psiddot_fin = 0.0

	psi_des = psi_init

	x_des_traj_init = x_init
	y_des_traj_init = y_init

	vx_des = 1.0
	vy_des = 0.0

	############################################################## Hyperparameters
	t_fin = 30
	lx = 2.0
	a_obs_elliptical = 0.70+0.10 ######### size of the obstacle inflated by the radius of each-circle of the robot

	### 0.70 is the obstacle+robot radius, 0.10 is the inflation to account for the fact that optimizer residual will not be exactly zero

	maxiter_global = 10
	mpc_iter = 800

	weight_smoothness = 10**2
	rho_ineq = 1.0
	rho_obs_elliptical = 1
	rho_w = 1
	rho_psi =1
	weight_psi = 1.0
	rho_track = 1.0

	############################################################################

	prob = batch_mpc_planner.Traj_Opt_Elliptical(num_goal, psi_des, t_fin, lx, a_obs_elliptical, maxiter_global, weight_smoothness, rho_ineq, rho_obs_elliptical, rho_w, rho_psi, weight_psi, rho_track)
	prob.jit_compile()


	############# Reading obstacle instantaneous position and velocity from the file
	x_obs_track = jnp.asarray(data['x_obs_track']).T
	y_obs_track = jnp.asarray(data['y_obs_track']).T


	vx_obs_track = jnp.asarray(data['vx_obs_track'])
	vy_obs_track = jnp.asarray(data['vy_obs_track'])

	######################################## noise sampling

	########### Random samples for batch initialization of heading angles
	psi_noise = np.random.normal(np.zeros(prob.num), 0.3, (prob.num_goal, 100))



	A = np.diff(np.diff(np.identity(prob.num), axis = 0), axis = 0)
	# A = np.diff(np.identity(prob.num), axis =0 )

	temp_1 = np.zeros(prob.num)
	temp_2 = np.zeros(prob.num)
	temp_3 = np.zeros(prob.num)
	temp_4 = np.zeros(prob.num)

	temp_1[0] = 1.0
	temp_2[0] = -2
	temp_2[1] = 1
	temp_3[-1] = -2
	temp_3[-2] = 1

	temp_4[-1] = 1.0

	A_mat = -np.vstack(( temp_1, temp_2, A, temp_3, temp_4   ))
	# A_mat = A

	R = np.dot(A_mat.T, A_mat)
	mu = np.zeros(prob.num)
	cov = np.linalg.pinv(R)

	################# Gaussian Trajectory Sampling
	eps_k = np.random.multivariate_normal(mu, 0.03*cov, (prob.num_goal, ))


	############################ Arrays for storing trajectory values during MPC run
	x_robot = np.zeros(mpc_iter)
	y_robot = np.zeros(mpc_iter)
	psi_robot = np.zeros(mpc_iter)

	vx_robot = np.zeros(mpc_iter)
	vy_robot = np.zeros(mpc_iter)
	psidot_robot = np.zeros(mpc_iter)


	ax_robot = np.zeros(mpc_iter)
	ay_robot = np.zeros(mpc_iter)
	psiddot_robot = np.zeros(mpc_iter)

	x_des_robot = np.zeros(mpc_iter)
	y_des_robot = np.zeros(mpc_iter)

	res_obs_robot = np.zeros(mpc_iter)


	t_update = 0.08 #### Delta t in state update
	t_control = 0.02 #### upsampling
	num_up = int(prob.t_fin/t_control)
	# num_up = prob_data.num

	tot_time_up = np.linspace(0.0, prob.t_fin, num_up)
	tot_time_copy_up = tot_time_up.reshape(num_up, 1)
	P_up, Pdot_up, Pddot_up = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, tot_time_copy_up[0], tot_time_copy_up[-1], tot_time_copy_up)


	x_homotopies = []
	y_homotopies = []


	x_des_pred = []
	y_des_pred = []

	x_best = []
	y_best = []


	best_index = np.zeros(mpc_iter)


	for i in range(0, mpc_iter):

		start = time.time()
		sol_x_np, sol_y_np, sol_psi_np, idx_np, res_track_np, idx_obs_np, res_obs_np = batch_mpc_planner.compute_mpc(prob, i, x_init, vx_init, ax_init, y_init, vy_init, ay_init, x_fin, y_fin, psi_init, psidot_init, psiddot_init, jnp.asarray(x_obs_track), jnp.asarray(y_obs_track), jnp.asarray(vx_obs_track), jnp.asarray(vy_obs_track), psi_noise, eps_k, vx_des, vy_des, x_des_traj_init, y_des_traj_init)

		idx_unique = idx_np[idx_np!=0]

		
		if(np.shape(idx_unique)[0]>0):
			res_index_temp = np.argmin(res_track_np[idx_unique])
			res_index = idx_unique[res_index_temp]

		if(np.shape(idx_unique)[0]==0):

			res_index = idx_obs_np
			idx_unique = idx_obs_np

		best_index[i] = res_index	

		sol_x_elite = sol_x_np[res_index]	
		sol_y_elite = sol_y_np[res_index]	
		sol_psi_elite = sol_psi_np[res_index]	

		# x_best.append(np.dot(prob.P, sol_x_elite))
		# y_best.append(np.dot(prob.P, sol_y_elite))

		x_feas = np.dot(prob.P, sol_x_np[idx_unique].T).T
		y_feas = np.dot(prob.P, sol_y_np[idx_unique].T).T

		####### uncomment this if you want to store all the feasible homotopies computed at each MPC loop

		# x_homotopies.append(x_feas)
		# y_homotopies.append(y_feas)

		xddot_ellite = np.dot(Pddot_up, sol_x_elite)
		yddot_ellite = np.dot(Pddot_up, sol_y_elite)
		psiddot_ellite = np.dot(Pddot_up, sol_psi_elite)

		xdot_ellite = np.dot(Pdot_up, sol_x_elite)
		ydot_ellite = np.dot(Pdot_up, sol_y_elite)
		psidot_ellite = np.dot(Pdot_up, sol_psi_elite)

		
	 ############## computing acceleration and velocities over a short interval [0, 0.02] and then averaging to get a piece-wise constant value
		ax_init = np.mean(xddot_ellite[0:10])
		ay_init = np.mean(yddot_ellite[0:10])
		psiddot_init = np.mean(psiddot_ellite[0:10])

		vx_init = np.mean(xdot_ellite[0:10])
		vy_init = np.mean(ydot_ellite[0:10])
		psidot_init = np.mean(psidot_ellite[0:10])
		


		x_init = x_init+vx_init*t_update
		y_init = y_init+vy_init*t_update
		psi_init = psi_init+psidot_init*t_update

		x_des_traj_init = x_des_traj_init+vx_des*t_update
		y_des_traj_init = y_des_traj_init+vy_des*t_update

		# x_des_pred.append(x_des_traj_init+vx_des*prob.tot_time)
		# y_des_pred.append(y_des_traj_init+vy_des*prob.tot_time)


		x_robot[i] = x_init
		y_robot[i] = y_init
		psi_robot[i] = psi_init
		
		vx_robot[i] = vx_init
		vy_robot[i] = vy_init
		psidot_robot[i] = psidot_init

		ax_robot[i] = ax_init
		ay_robot[i] = ay_init
		psiddot_robot[i] = psiddot_init

		x_des_robot[i] = x_des_traj_init
		y_des_robot[i] = y_des_traj_init

		res_obs_robot[i] = res_obs_np[res_index]

		# print(res_obs_np[res_index])
		

		print('mpc comp.time = ', time.time()-start)

# /home/arun/Batch_Traj_Opt_Holonomic/results_ours

	# obstacle_data_1 results gets stored in config_1 and so on

	file_name_temp = 'results_ours/'+'config_1/'+'results_dynamic_num'+str(num_goal)+'_'+str(k+1)+'.mat'	


	# file_name_temp = '/home/aks/video_data_icra_22/same_direction_ours/'+'results_dynamic_num'+str(num_goal)+'_'+str(k+1)+'.mat'	

	scipy.io.savemat(file_name_temp, {'x_robot': x_robot, 'y_robot': y_robot, 'psi_robot': psi_robot, 'x_obs_track':x_obs_track, 'y_obs_track':y_obs_track, 'x_des_robot':x_des_robot, 'y_des_robot':y_des_robot, 'vx_robot':vx_robot, 'vy_robot':vy_robot, 'ax_robot':ax_robot, 'ay_robot':ay_robot, 'psidot_robot':psidot_robot, 'psiddot_robot':psiddot_robot   }) ########## matrix of y position of the vehicle



# dist = compute_min_dist(x_robot, y_robot, x_obs_track, y_obs_track, psi_robot, lx)

# print('mindist =', np.amin(dist))


# x_feas = np.dot(prob.P, sol_x_np[idx_unique].T).T
# y_feas = np.dot(prob.P, sol_y_np[idx_unique].T).T
# psi_feas = np.dot(prob.P, sol_psi_np[idx_unique].T).T


# th = np.linspace(0, 2*np.pi, 100)

# plt.figure(1)
# plt.plot(x_feas.T, y_feas.T, '-m', linewidth = 2.0)
# plt.plot(x_robot[0:i], y_robot[0:i], '-k', linewidth = 3.0)
# plt.plot(x_des_robot, y_des_robot, '-g', linewidth = 5.0)
# # plt.plot(x_des_pred[-1], y_des_pred[-1], '-r', linewidth = 3.0)
# # plt.plot(x_elliptical[res_index].T, y_elliptical[res_index].T, '-k', linewidth = 3.0)
# # plt.plot(x_samples.T, y_samples.T, '-b', linewidth = 0.1)
# for j in range(0, prob.num_obs):
        
#     x_circ_1 = x_obs_track[j,mpc_iter]+(prob.a_obs_elliptical)*np.cos(th)
#     y_circ_1 = y_obs_track[j,mpc_iter]+(prob.a_obs_elliptical)*np.sin(th)
#     plt.plot(x_circ_1, y_circ_1, '-r', linewidth = 3.0)

# plt.axis('equal')



# plt.figure(2)
# plt.plot(np.sqrt(vx_robot**2+vy_robot**2))
# # plt.plot(vy_robot[0:i])

# plt.figure(3)
# plt.plot(psi_robot)

# plt.figure(4)
# plt.plot(np.sqrt(ax_robot**2+ay_robot**2))

# plt.figure(5)
# plt.plot(res_obs_robot)


# plt.show()	


