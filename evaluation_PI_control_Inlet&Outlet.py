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
# 2: Inlet  Boundary Control [ONLY]


control_settings['Scenario'] = 3
print('-------------------------------------')
print('-------------------------------------')
print('Evaluating Openloop Control Performance of Case {}.'.format(control_settings['Scenario']))
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
T = settings['T']
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
# Openloop Controller Setting
##########################################################################################
# Simulation setting

r_save_base = np.zeros([r_size,N])
y_save_base = np.zeros([y_size,N])
v_save_base = np.zeros([v_size,N])

#if control_settings['Scenario'] = 3:
#    action_save_base = np.zeros([2,N])
#else:
#    action_save_base = np.zeros([1,N])

action_save_base = np.zeros([2,N])

reward_save_base = np.zeros(N)

env.reset()
r_save_base[:,0] = env.r.reshape(r_size,)
y_save_base[:,0] = env.y.reshape(y_size,)
v_save_base[:,0] = env.v.reshape(v_size,)
action_save_base[:,0] = 0



rs = env.rs#0.12
vs = env.vs#10
qs = env.qs

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

# Find input index close to qs_input
env_qs_input = env.qs_input
idx = find_nearest(env_qs_input,qs)

DISCRETE = env.discrete


# PI Controller Settings
k_iv = 0.001
k_pv = -0.16
k_ir = -0.01
k_pr = -1.6

qs_inlet = qs #inital qs_inlet input.
qs_outlet = qs



for i in range(N-1):
    #action = find_nearest(env_qs_input,qs)
    if DISCRETE:
        action = find_nearest(env_qs_input,qs)
    else:
        action = np.array([qs_inlet, qs_outlet])
        action = action.reshape(2,) # Reshaping inputs.
    
    states, reward, is_done, info = env.step(action)
    r_save_base[:,i+1] = env.r.reshape(r_size,)
    y_save_base[:,i+1] = env.y.reshape(y_size,)
    v_save_base[:,i+1] = env.v.reshape(v_size,)

    # P controller Setting
    # v_temp = y_save_base[1,i]/r_save_base[1,i+1] + Veq(r_save_base[1,i+1])
    # U_in = (rs - qs/(gam * ps)) * (v_temp -vs)
    # qs_inlet = qs + U_in

    # PI Controller settings
    ## Inlet 
    tt = np.arange(0,dt*i+dt,dt)
    rr = r_save_base[-1,0:i+1] - rs*np.ones(i+1)
    q_in = k_pr * (r_save_base[-1,i+1] - rs) + k_ir*np.trapz(tt,rr)
    qs_inlet = qs + q_in

    ## Outlet
    v_temp = y_save_base[1,i]/r_save_base[1,i+1] + Veq(r_save_base[1,i+1])
    vv = v_temp - vs*np.ones(i+1)
    U_n = k_pv * (v_temp - vs) + k_iv * np.trapz(tt,vv)
    qs_outlet = r_save_base[-1,i+1] *(U_n + vs)

    

    action_save_base[:,i+1] = action
    reward_save_base[i+1] = reward



##########################################################################################
# Visualization
##########################################################################################
print('-------------------------------------')
print('Visualization')
print('-------------------------------------')


if control_settings['Scenario'] == 3:
    print('Plotting Outlet & Inlet Openloop (Setpoint control) Case.')
    ## create meshgrid
    L = settings['L']  #[m]
    dx = settings['dx']
    x = np.arange(0,L+dx,dx)
    t = np.arange(0,T+dt,dt)

    xx, tt = np.meshgrid(x,t,indexing='ij')

    # Action
    fig, ax1 = plt.subplots(figsize=(6,4))
    U_inlet = r_save_base[0,1:] * v_save_base[0,1:]
    U_outlet = r_save_base[-1,1:] * v_save_base[-1,1:]
    p1=plt.plot(U_inlet*3600,'b',label='Inlet Openloop control')
    p2=plt.plot(U_outlet*3600,'b-.',label='Outlet Openloop control')
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
    ax.plot_surface(xx[0:,0:],tt[0:,0:],r_save_base[0:,0:],cmap=plt.cm.gray,edgecolors='#000000',linewidth=0.01,antialiased=False,rstride=1,cstride=100)
    ax.plot(xx[:,0],tt[:,0],r_save_base[:,0],color='blue',LineWidth=4)
    ax.plot(xx[0,:],tt[0,:],r_save_base[0,:],color='red',LineWidth=4)
    ax.plot(xx[-1,:],tt[-1,:],r_save_base[-1,:],color='red',LineWidth=4)
    ax.invert_yaxis()
    plt.show()

    # Velocity
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Position x [m]')
    ax.set_ylabel('Time [min]')
    ax.set_zlabel('Velocity [m/s]')
    ax.set_title(r'$V$: Velocity')
    ax.plot_surface(xx[0:,0:],tt[0:,0:],v_save_base[0:,0:],cmap=plt.cm.gray,edgecolors='#000000',linewidth=0.01,antialiased=False,rstride=1,cstride=100)
    ax.plot(xx[:,0],tt[:,0],v_save_base[:,0],color='blue',LineWidth=4)
    ax.plot(xx[0,:],tt[0,:],v_save_base[0,:],color='red',LineWidth=4)
    ax.plot(xx[-1,:],tt[-1,:],v_save_base[-1,:],color='red',LineWidth=4)
    ax.invert_yaxis()
    plt.show()

    # Reward
    fig, ax1 = plt.subplots(figsize=(8,4))

    rwd_save_base = reward_save_base

    plt.plot(t[1:],rwd_save_base[1:])
    plt.xlabel('Time [sec]')
    plt.ylabel('Reward')
    plt.title('Reward')
    plt.grid(True)
    plt.show()
else:
    raise ValueError('This is only for PI control validation in Inlet&Outlet control.')




##########################################################################################
# SAVE
##########################################################################################
print('-------------------------------------')
print('Save the data as .mat format')
print('-------------------------------------')
if control_settings['Scenario'] == 3:
   sio.savemat('save_mat/ARZ_PI_Control_Outlet_N_Inlet_Results.mat',{'r_vec_base':r_save_base, 'v_vec_base':v_save_base, 'rwd_base' : reward_save_base, 'input_base': action_save_base, 'xx': xx, 'tt' : tt})
   print('Saved')
