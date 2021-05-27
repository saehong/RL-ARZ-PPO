# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 02:19:27 2020

@author: Saehong Park
"""

# TRAINING settings
 #settings 
settings={}

settings['vm']=40 #[m/s], maximum value for velocity(v)
settings['rm']=0.16 #[veh/m], maximum value for rho(r)
settings['qm']=settings['vm']*settings['rm']/4
settings['L']=500 #[m]
settings['tau'] = 60 # [?]
settings['T'] = 240 #[sec]

settings['vs'] = 10 #[m/s]
settings['vs_desired'] = 10

settings['rs'] = 0.12
settings['rs_desired'] = 0.12

settings['qs'] = settings['rs']*settings['vs']
settings['qs_desired'] = settings['rs_desired'] * settings['vs_desired']

settings['ps'] = settings['vm']/settings['rm'] * settings['qs']/settings['vs']
settings['ys'] = 0

settings['gam'] = 1

settings['T'] = 120
settings['dt'] = 0.1 
settings['dx'] = 2.5




# # Parameters
# vm = 40 
# rm = 0.16 #[veh/m] # maximum value for rho(r)
# qm = vm*rm/4 # maximum value for q
# # tau = 60 #[?] 
# L = 500  #[m]
# #		T = 240 #[sec]
# ### vs = 10 #[m/s]
# vs_desired = 10
# ### rs = 0.12 #[veh/m] // Randomize
# rs = 0.12 
# rs_desired = 0.12 
# ### qs = rs * vs #[veh/s] // randomized rs
# qs_desired = rs_desired * vs_desired

# gam = 1
# ### ps = vm/rm * qs/vs #// randomized rs
# ys = 0

# # Discretization
# dx = 5
# dt = 0.1
# #		t = np.arange(0,T+dt,dt)
# x = np.arange(0,L+dx,dx)
# M = len(x)

