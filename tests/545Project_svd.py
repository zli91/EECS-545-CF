# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 22:06:29 2018

@author: HUAWEI
"""

import numpy as np
import surprise as sp

# u_id:user id      i_id:item id
# un_: user number  in_: item number
# real rating matrix: r[u_id][i_id]
# estimate : r_h[m][u_id][i_id]
# ###########similarity matrix: (item_based) s[i_id][i_id]
# rating interval: 
# weight matrix: W[u_id][i_id]
# k_neighbor: k_neighbor_li

D = 5-1
# calculate denominator
w_sum = 0
for u in range(un_):
    for i in range(in_):
        w_sum += W[u_id][i_id]

# calculate errRate
errRate = 0
for u in range(un_):
    for i in range(in_):        
        abs_err = abs(r[u_id][i_id]-r_h[m][u_id][i_id])
                       
        errRate += W[u_id][i_id]*abs_err/D/w_sum
        
# UE[u_id][i_id]
# m = current classifier
yita = 0.5

for u in range(un_):
    for i in range(in_): 
        UE[u_id][i_id] = 1 + yita*r[u_id][i_id]/D
        for mm in range(m):
            UE[u_id][i_id] -= yita*r_h[mm][u_id][i_id]/m/D   
            
# Update weights
# rho belongs to [0.2,0.3,0.4,0.5,0.6]
for u in range(un_):
    for i in range(in_):
        for k_id in k_neighbor_li:
            W[u_id][k_id] = 1 - errRate/(1-errRate)*UE[u_id][k_id]*rho

#Normalize
W = W/np.sum(W)*un_*in_

#Prediction
# At the very beginning, set:
Pred = 0

# recommender weight : recm_w
recm_w = np.log(1/errRate-1)

Pred += r_h[m]*recm_w