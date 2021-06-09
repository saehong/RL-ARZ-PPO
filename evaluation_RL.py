import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gym
import gym_arz
from settings_file import *
import argparse
import os
import sys

import numpy as np
import torch

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs, make_vec_envs_arz
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

from types import SimpleNamespace

import scipy.io as sio
import ipdb
##########################################################################################
# Case
##########################################################################################
## Cases
# 1: Outlet Boundary Control
# 2: Inlet  Boundary Control
# 3: Outlet & Inlet Boundary Control
control_settings['Scenario'] = 3
print('-------------------------------------')
print('-------------------------------------')
print('Evaluating RL Performance of Case {}.'.format(control_settings['Scenario']))
print('-------------------------------------')
print('-------------------------------------')


##########################################################################################
# Environment Setting
##########################################################################################

# Load Environment.
env = gym.make("arz-v0", sett=settings, cont_sett = control_settings)


# Env. Setting
r_size = len(env.r)
y_size = len(env.y)
v_size = len(env.v)

## Parameter values from the Gym environment
# Parameter
vm = settings['vm']
rm = settings['rm']
tau = settings['tau']
L = settings['L']
T = settings['T'] * 4 
vs = settings['vs']
rs  = settings['rs']
qs  = settings['qs']
gam = settings['gam']
ps  = vm/rm * qs/vs
ys  = 0;


# Discretization
dx = settings['dx']
dt = settings['dt']
t = np.arange(0,T+dt,dt)
x = np.arange(0,L+dx,dx)
M = len(x)
N = len(np.arange(0,T+dt,dt))

# charateristics
lambda_1 = vs ;
lambda_2 = vs - rs * vm/rm ;

# Fundamental diagram
Veq = lambda rho: vm * ( 1 - rho/rm)

# Flux
F_r = lambda rho,y: y + rho * Veq(rho)
F_y = lambda rho,y: y * (y/rho + Veq(rho))

# Spatial function
c_x = lambda x: -1 / tau * np.exp(-x/tau/vs)

##########################################################################################
# RL Controller Setting
##########################################################################################

## 1.Control Outlet, fix inlet
if control_settings['Scenario'] == 1:
    args = SimpleNamespace(env_name="arz-v0", load_dir="save_trained_results/1_Outlet_Boundary_Results", seed = 1, det = True, non_det = False, log_interval=10)

## 2.Control inlet, fix outlet 
if control_settings['Scenario'] == 2:
    args = SimpleNamespace(env_name="arz-v0", load_dir="save_trained_results/2_Inlet_Boundary_Results", seed = 1, det = True, non_det = False, log_interval=10)

## 3.Control inlet & outlet 
if control_settings['Scenario'] == 3:
    args = SimpleNamespace(env_name="arz-v0", load_dir="save_trained_results/3_Outlet_Inlet_Boundary_Results", seed = 1, det = True, non_det = False, log_interval=10)

## 4.Control inlet & outlet w/ random r_s
# args = SimpleNamespace(env_name="arz-v0", load_dir="./0917_inlet_outlet_controls_random/2020-09-17-12-06", seed = 1, det = True, non_det = False, log_interval=10)


env_vec = make_vec_envs_arz(
    args.env_name,
    settings,
    control_settings,
    args.seed + 1000,
    1,
    None,
    None,
    device='cpu',
    allow_early_resets=False)


# Get a render function
render_func = get_render_func(env_vec)

# Choose the trained_network model
## 1.Control Outlet, fix inlet
if control_settings['Scenario'] == 1:
    actor_critic, ob_rms = \
                torch.load(os.path.join(args.load_dir, args.env_name + "-tr-1040-th" + ".pt"),map_location='cpu')

## 2.Control inlet, fix outlet controller
if control_settings['Scenario'] == 2:
    actor_critic, ob_rms = \
                torch.load(os.path.join(args.load_dir, args.env_name + "-tr-1040-th" + ".pt"),map_location='cpu')

## 3.Control inlet & outlet
if control_settings['Scenario'] == 3:
    actor_critic, ob_rms = \
                torch.load(os.path.join(args.load_dir, args.env_name + "-tr-1040-th" + ".pt"),map_location='cpu')

## 4.Control inlet & outlet w/ random r_s
# actor_critic, ob_rms = \
#             torch.load(os.path.join(args.load_dir, args.env_name + "-tr-2080-th" + ".pt"),map_location='cpu')


vec_norm = get_vec_normalize(env_vec)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1,
                                      actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

obs = env_vec.reset()


##########################################################################################
#Simulation
##########################################################################################
print('-------------------------------------')
print('Evaluating Simulation')
print('-------------------------------------')
obs_save = []
rwd_save = []
PPO_input_save = []
PPO_RHO_SAVE = []
PPO_VEL_SAVE = []

TIME_VEC = []
dt = settings['dt']

# Intial Condtions
obs = env_vec.reset()
obs_save.append(obs)

PPO_RHO_SAVE.append((obs[0][0:51]*env.rs_desired+env.rs_desired).tolist())
PPO_VEL_SAVE.append((obs[0][51:]*env.vs_desired+env.vs_desired).tolist())


ACTION_VEC = []
if control_settings['Scenario'] == 1:
    ACTION_VEC.append(np.array([0]))

if control_settings['Scenario'] == 2:
    ACTION_VEC.append(np.array([0]))

if control_settings['Scenario'] == 3:
    ACTION_VEC.append(np.array([0,0]))


tt = 0
TIME_VEC.append(tt)




# while True:
for i in range(N-1):
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)
        
    action = action[0]

    PPO_input_save.append(action)
    
    obs, reward, done, _ = env_vec.step(action)
    
    
    if done:
        print('DONE')
        break
    
    ## Append Observation & Reward
    obs_save.append(obs) 
    rwd_save.append(reward) 
              
    ## Vectorized version
    PPO_RHO_SAVE.append((obs[0][0:51]*env.rs_desired+env.rs_desired).tolist())
    PPO_VEL_SAVE.append((obs[0][51:]*env.vs_desired+env.vs_desired).tolist())
    
    ## Append Action
    ACTION_VEC.append(action.detach().numpy())



        
    ## 
    tt += dt
    TIME_VEC.append(tt)
    


##########################################################################################
# Visualization
##########################################################################################
print('-------------------------------------')
print('Visualization')
print('-------------------------------------')
# Plot
r_save_RL = np.array(PPO_RHO_SAVE)
r_save_RL = r_save_RL.T

v_save_RL = np.array(PPO_VEL_SAVE)
v_save_RL = v_save_RL.T

action_save_RL = np.array(ACTION_VEC)
reward_save_RL = rwd_save    


if control_settings['Scenario'] == 1:
    print('Plotting Outlet RL Case.')
    ## create meshgrid
    L = settings['L']  #[m]
    dx = settings['dx']
    x = np.arange(0,L+dx,dx)
    t = np.arange(0,T,dt)

    xx, tt = np.meshgrid(x,t,indexing='ij')

    # Action
    fig, ax1 = plt.subplots(figsize=(6,4))
    U_outlet = r_save_RL[-1,1:] * v_save_RL[-1,1:]
    plt.plot(U_outlet*3600);
    plt.xlabel('Time [sec]')
    plt.ylabel(r'$U_{out}(t)$ [veh/h]')
    plt.title('Outlet INPUT SHAPE')
    plt.grid(True)
    plt.show()


    # Density
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Position x [m]')
    ax.set_ylabel('Time [min]')
    ax.set_zlabel('Density [veh/km]')
    ax.set_title(r'$\rho$: Density')
    ax.plot_surface(xx[0:,0:],tt[0:,0:],r_save_RL[0:,0:],cmap=plt.cm.gray,edgecolors='#000000',linewidth=0.01,antialiased=False,rstride=1,cstride=100)
    ax.plot(xx[:,0],tt[:,0],r_save_RL[:,0],color='blue',LineWidth=4)
    ax.plot(xx[-1,:],tt[-1,:],r_save_RL[-1,:],color='red',LineWidth=4)
    plt.show()

    # Velocity
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Position x [m]')
    ax.set_ylabel('Time [min]')
    ax.set_zlabel('Velocity [m/s]')
    ax.set_title(r'$V$: Velocity')
    ax.plot_surface(xx[0:,0:],tt[0:,0:],v_save_RL[0:,0:],cmap=plt.cm.gray,edgecolors='#000000',linewidth=0.01,antialiased=False,rstride=1,cstride=100)
    ax.plot(xx[:,0],tt[:,0],v_save_RL[:,0],color='blue',LineWidth=4)
    ax.plot(xx[-1,:],tt[-1,:],v_save_RL[-1,:],color='red',LineWidth=4)
    plt.show()

    # Reward
    fig, ax1 = plt.subplots(figsize=(8,4))

    rwd_save_PPO = np.array(reward_save_RL)

    plt.plot(TIME_VEC[:-1],reward_save_RL)
    plt.xlabel('Time [sec]')
    plt.ylabel('Reward')
    plt.title('Reward')
    plt.grid(True)
    plt.show()



if control_settings['Scenario'] == 2:
    print('Plotting Inlet RL Case.')
    ## create meshgrid
    L = settings['L']  #[m]
    dx = settings['dx']
    x = np.arange(0,L+dx,dx)
    t = np.arange(0,T,dt)

    xx, tt = np.meshgrid(x,t,indexing='ij')

    # Action
    fig, ax1 = plt.subplots(figsize=(6,4))
    U_inlet = r_save_RL[0,1:] * v_save_RL[0,1:]
    plt.plot(U_inlet*3600)
    plt.xlabel('Time [sec]')
    plt.ylabel(r'$U_{in}(t)$ [veh/h]')
    plt.title('Inlet INPUT SHAPE')
    plt.grid(True)
    plt.show()

    # Density
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Position x [m]')
    ax.set_ylabel('Time [min]')
    ax.set_zlabel('Density [veh/km]')
    ax.set_title(r'$\rho$: Density')
    ax.plot_surface(xx[0:,0:],tt[0:,0:],r_save_RL[0:,0:],cmap=plt.cm.gray,edgecolors='#000000',linewidth=0.01,antialiased=False,rstride=1,cstride=100)
    ax.plot(xx[:,0],tt[:,0],r_save_RL[:,0],color='blue',LineWidth=4)
    ax.plot(xx[0,:],tt[0,:],r_save_RL[0,:],color='red',LineWidth=4)
    ax.invert_yaxis()
    plt.show()

    # Velocity
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Position x [m]')
    ax.set_ylabel('Time [min]')
    ax.set_zlabel('Velocity [m/s]')
    ax.set_title(r'$V$: Velocity')
    ax.plot_surface(xx[0:,0:],tt[0:,0:],v_save_RL[0:,0:],cmap=plt.cm.gray,edgecolors='#000000',linewidth=0.01,antialiased=False,rstride=1,cstride=100)
    ax.plot(xx[:,0],tt[:,0],v_save_RL[:,0],color='blue',LineWidth=4)
    ax.plot(xx[0,:],tt[0,:],v_save_RL[0,:],color='red',LineWidth=4)
    ax.invert_yaxis()
    plt.show()

    # Reward
    fig, ax1 = plt.subplots(figsize=(8,4))

    rwd_save_PPO = np.array(reward_save_RL)

    plt.plot(TIME_VEC[:-1],reward_save_RL)
    plt.xlabel('Time [sec]')
    plt.ylabel('Reward')
    plt.title('Reward')
    plt.grid(True)
    plt.show()


if control_settings['Scenario'] == 3:
    print('Plotting Outlet & Inlet RL Case.')
    ## create meshgrid
    L = settings['L']  #[m]
    dx = settings['dx']
    x = np.arange(0,L+dx,dx)
    t = np.arange(0,T,dt)

    xx, tt = np.meshgrid(x,t,indexing='ij')

    # Action
    fig, ax1 = plt.subplots(figsize=(6,4))
    U_inlet = r_save_RL[0,1:] * v_save_RL[0,1:]
    U_outlet = r_save_RL[-1,1:] * v_save_RL[-1,1:]
    p1=plt.plot(U_inlet*3600,'b',label='Inlet RL control')
    p2=plt.plot(U_outlet*3600,'b-.',label='Outlet RL control')
    plt.xlabel('Time [sec]')
    plt.ylabel(r'$U_{in,out}(t)$ [veh/h]')
    plt.title('Inlet & Outlet INPUT SHAPE')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Density
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Position x [m]')
    ax.set_ylabel('Time [min]')
    ax.set_zlabel('Density [veh/km]')
    ax.set_title(r'$\rho$: Density')
    ax.plot_surface(xx[0:,0:],tt[0:,0:],r_save_RL[0:,0:],cmap=plt.cm.gray,edgecolors='#000000',linewidth=0.01,antialiased=False,rstride=1,cstride=100)
    ax.plot(xx[:,0],tt[:,0],r_save_RL[:,0],color='blue',LineWidth=4)
    ax.plot(xx[0,:],tt[0,:],r_save_RL[0,:],color='red',LineWidth=4)
    ax.plot(xx[-1,:],tt[-1,:],r_save_RL[-1,:],color='red',LineWidth=4)
    ax.invert_yaxis()
    plt.show()

    # Velocity
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Position x [m]')
    ax.set_ylabel('Time [min]')
    ax.set_zlabel('Velocity [m/s]')
    ax.set_title(r'$V$: Velocity')
    ax.plot_surface(xx[0:,0:],tt[0:,0:],v_save_RL[0:,0:],cmap=plt.cm.gray,edgecolors='#000000',linewidth=0.01,antialiased=False,rstride=1,cstride=100)
    ax.plot(xx[:,0],tt[:,0],v_save_RL[:,0],color='blue',LineWidth=4)
    ax.plot(xx[0,:],tt[0,:],v_save_RL[0,:],color='red',LineWidth=4)
    ax.plot(xx[-1,:],tt[-1,:],v_save_RL[-1,:],color='red',LineWidth=4)
    ax.invert_yaxis()
    plt.show()

    # Reward
    fig, ax1 = plt.subplots(figsize=(8,4))

    rwd_save_PPO = np.array(reward_save_RL)

    plt.plot(TIME_VEC[:-1],reward_save_RL)
    plt.xlabel('Time [sec]')
    plt.ylabel('Reward')
    plt.title('Reward')
    plt.grid(True)
    plt.show()




##########################################################################################
# SAVE
##########################################################################################
print('-------------------------------------')
print('Save the data as .mat format')
print('-------------------------------------')
if control_settings['Scenario'] == 1:
    sio.savemat('save_mat/ARZ_RL_Outlet_Results.mat',{'r_vec_RL':r_save_RL, 'v_vec_RL':v_save_RL, 'rwd_RL' : reward_save_RL, 'input_RL': action_save_RL, 'xx': xx, 'tt' : tt})
    print('Saved')
elif control_settings['Scenario'] == 2:
    sio.savemat('save_mat/ARZ_RL_Inlet_Results.mat',{'r_vec_RL':r_save_RL, 'v_vec_RL':v_save_RL, 'rwd_RL' : reward_save_RL, 'input_RL': action_save_RL, 'xx': xx, 'tt' : tt})
    print('Saved')
elif control_settings['Scenario'] == 3:
    sio.savemat('save_mat/ARZ_RL_Outlet_N_Inlet_Results.mat',{'r_vec_RL':r_save_RL, 'v_vec_RL':v_save_RL, 'rwd_RL' : reward_save_RL, 'input_RL': action_save_RL, 'xx': xx, 'tt' : tt})
    print('Saved')
