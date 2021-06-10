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

settings['T'] = 240 # 4 min
settings['dt'] = 0.25 #0.25/50*60 (M=60) #0.25 (M=50) #0.25/50*40 (M=40)
settings['dx'] = 10 #500/60 (M=60) #10 (M=50) #500/40 (M=40)

print('Settins of dt:{}'.format(settings['dt'] ))
print('Settins of dx:{}'.format(settings['dx'] ))

control_settings={}

## Cases
# 1: Outlet Boundary Control
# 2: Inlet  Boundary Control
# 3: Outlet & Inlet Boundary Control

choose_case = 1
control_settings['Scenario'] = choose_case