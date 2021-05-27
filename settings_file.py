# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 02:19:27 2020

@author: Saehong Park
"""

## ARZ Model Parameters
# TRAINING settings
settings={}

settings['vm']=40 #[m/s], maximum value for velocity(v)
settings['rm']=0.16 #[veh/m], maximum value for rho(r)
settings['qm']=settings['vm']*settings['rm']/4
settings['L']=500 #[m]
settings['tau'] = 60 # [?]

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
settings['dt'] = 0.25 #0.1 
settings['dx'] = 10 #2.5


control_settings={}

