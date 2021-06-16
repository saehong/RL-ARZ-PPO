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


control_settings['Scenario'] = 1
print('-------------------------------------')
print('-------------------------------------')
print('Evaluating Backstepping Control Performance of Outlet Case {}.'.format(control_settings['Scenario']))
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
# Backstepping Controller Setting
##########################################################################################

# Kernels K and L
K = np.zeros((M,M))

K[0,0] = 1/tau/(gam*ps)
K[1,0] = K[0,0] + vs / (gam * ps - vs) * (K[0,1]-K[0,0]) + dx * (-1 / tau) * np.exp(-(1-1) * dx/(tau*vs)) / (gam * ps - vs) * K[0,0]

for z in range(1,M-1) :
    K[z,z] = 1/tau * np.exp(-(z) * dx/(tau*vs))/(gam*ps)
    #import ipdb; ipdb.set_trace()
    K[z+1,0] = K[z,0] + vs / (gam * ps - vs) * (K[z,1]-2*K[z,0])/2 + dx * (-1/tau) * np.exp(-(1-1) * dx/(tau*vs))/(gam * ps - vs) * K[z,0]
    for xi in range(1,z+1) :
        K[z+1,xi] = K[z, xi] + vs / (gam * ps - vs) * (K[z,xi+1]-2*K[z,xi]+K[z,xi-1])/2 + dx * (-1/tau) * np.exp(-(xi) * dx/(tau*vs))/(gam * ps - vs) * K[z-xi,0]

K[M-1,M-1] = 1/tau * np.exp(-(M-1) * dx/(tau*vs))/(gam*ps)


# Simulation setting

q_inlet_save = np.zeros([1,N])
r_save = np.zeros([r_size,N])
y_save = np.zeros([y_size,N])
v_save = np.zeros([v_size,N])
PDE_input_save = np.zeros([1,N])
reward_save = np.zeros(N)

r_save[:,0] = env.r.reshape(r_size,)
y_save[:,0] = env.y.reshape(y_size,)
v_save[:,0] = env.v.reshape(v_size,)

v_temp = y_save[:,0]/r_save[:,0] + Veq(r_save[:,0]) - vs * np.ones((M,))
v_temp = v_temp.reshape(v_temp.shape[0],1) # flatten -> column vector
q_temp = y_save[:,0] + r_save[:,0] * Veq(r_save[:,0]) - qs * np.ones((M,))
q_temp = q_temp.reshape(q_temp.shape[0],1) # flatten -> column vector

Trans_K =  np.transpose(K[:,0]).reshape(1,K[:,0].shape[0])
IM = np.fliplr(-Trans_K) * np.transpose(v_temp)

IK_v_left = lambda_2 / lambda_1 * K[M-1,:] * np.exp(x/tau/vs)
IK_v_left = IK_v_left.reshape(1,IK_v_left.shape[0]) # flatten -> row vector
IK_v = IK_v_left * np.transpose(v_temp)
IK_q_left = (lambda_1 - lambda_2) / qs * K[M-1,:] * np.exp(x/tau/vs)
IK_q_left = IK_q_left.reshape(1,IK_q_left.shape[0]) # flatten -> row vector
IK_q = IK_q_left * np.transpose(q_temp)

U = np.zeros(N)
U[0] = np.trapz(IM,x=x) + np.trapz(IK_v,x=x) + np.trapz(IK_q,x=x)

qs_input = (U[0]+vs)*r_save[-1,0]

action_save_bkst = np.zeros([1,N])
action_save_bkst[:,0] = qs_input

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

# Find input index close to qs_input
env_qs_input = env.qs_input
idx = find_nearest(env_qs_input,qs_input)
DISCRETE = False

for i in range(N-1):
    #action = find_nearest(env_qs_input,qs_input)
    #q_inlet_save[:,i+1] = env.q_inlet.reshape(1,)
    if DISCRETE:
        action = find_nearest(env_qs_input,qs)
    else:
        action = qs_input
        PDE_input_save[:,i] = qs_input
    states, reward, is_done, info = env.step(action)
    r_save[:,i+1] = env.r.reshape(r_size,)
    y_save[:,i+1] = env.y.reshape(y_size,)
    v_save[:,i+1] = env.v.reshape(v_size,)
    reward_save[i+1] = reward
    
    # Closed-loop control
    v_temp = y_save[:,i+1]/r_save[:,i+1] + Veq(r_save[:,i+1]) - vs * np.ones((M,))
    v_temp = v_temp.reshape(v_temp.shape[0],1) # flatten -> column vector
    q_temp = y_save[:,i+1] + r_save[:,i+1] * Veq(r_save[:,i+1]) - qs * np.ones((M,))
    q_temp = q_temp.reshape(q_temp.shape[0],1) # flatten -> column vector

    Trans_K =  np.transpose(K[:,0]).reshape(1,K[:,0].shape[0])
    IM = np.fliplr(-Trans_K) * np.transpose(v_temp)

    IK_v_left = lambda_2 / lambda_1 * K[M-1,:] * np.exp(x/tau/vs)
    IK_v_left = IK_v_left.reshape(1,IK_v_left.shape[0]) # flatten -> row vector
    IK_v = IK_v_left * np.transpose(v_temp)
    IK_q_left = (lambda_1 - lambda_2) / qs * K[M-1,:] * np.exp(x/tau/vs)
    IK_q_left = IK_q_left.reshape(1,IK_q_left.shape[0]) # flatten -> row vector
    IK_q = IK_q_left * np.transpose(q_temp)

    U[i+1] = np.trapz(IM,x=x) + np.trapz(IK_v,x=x) + np.trapz(IK_q,x=x)

    qs_input = (U[i+1]+vs)*r_save[-1,i+1]

    action_save_bkst[:,i+1] = qs_input



##########################################################################################
# Visualization
##########################################################################################
print('-------------------------------------')
print('Visualization')
print('-------------------------------------')

r_save_bcmk = r_save
y_save_bcmk = y_save
v_save_bcmk = v_save

reward_save_bcmk = reward_save
action_save_bcmk = action_save_bkst

if control_settings['Scenario'] == 1:
    print('Plotting Outlet Backstepping Control Case.')
    ## create meshgrid
    L = settings['L']  #[m]
    dx = settings['dx']
    x = np.arange(0,L+dx,dx)
    t = np.arange(0,T+dt,dt)

    xx, tt = np.meshgrid(x,t,indexing='ij')

    # Action
    fig, ax1 = plt.subplots(figsize=(6,4))
    U_outlet = r_save_bcmk[-1,1:] * v_save_bcmk[-1,1:]
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
    ax.plot_surface(xx[0:,0:],tt[0:,0:],r_save_bcmk[0:,0:],cmap=plt.cm.gray,edgecolors='#000000',linewidth=0.01,antialiased=False,rstride=1,cstride=100)
    ax.plot(xx[:,0],tt[:,0],r_save_bcmk[:,0],color='blue',LineWidth=4)
    ax.plot(xx[-1,:],tt[-1,:],r_save_bcmk[-1,:],color='red',LineWidth=4)
    plt.show()

    # Velocity
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Position x [m]')
    ax.set_ylabel('Time [min]')
    ax.set_zlabel('Velocity [m/s]')
    ax.set_title(r'$V$: Velocity')
    ax.plot_surface(xx[0:,0:],tt[0:,0:],v_save_bcmk[0:,0:],cmap=plt.cm.gray,edgecolors='#000000',linewidth=0.01,antialiased=False,rstride=1,cstride=100)
    ax.plot(xx[:,0],tt[:,0],v_save_bcmk[:,0],color='blue',LineWidth=4)
    ax.plot(xx[-1,:],tt[-1,:],v_save_bcmk[-1,:],color='red',LineWidth=4)
    plt.show()

    # Reward
    fig, ax1 = plt.subplots(figsize=(8,4))
    

    rwd_save_bcmk = reward_save_bcmk

    plt.plot(t[1:],rwd_save_bcmk[1:])
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
    sio.savemat('save_mat/ARZ_Backstep_Outlet_Results.mat',{'r_vec_bcmk':r_save_bcmk, 'v_vec_bcmk':v_save_bcmk, 'rwd_bcmk' : reward_save_bcmk, 'input_bcmk': action_save_bcmk, 'xx': xx, 'tt' : tt})
    print('Saved')
