# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 22:09:32 2018

@author: HUAWEI
"""

import numpy as np
import surprise
from surprise import KNNBaseline
from surprise import Dataset
from surprise import get_dataset_dir
from surprise.model_selection import train_test_split
import pandas as pd


K = 10


data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()
trash, testset = train_test_split(data, test_size=.2)
# sim_options = {'name': 'pearson_baseline', 'user_based': False}

sim_options = {'user_based': False}
algo = KNNBaseline(sim_options=sim_options)
# algo = KNNBaseline()


D = 5-1
s = (trainset.n_users+1,trainset.n_items+1)
W = np.ones(s)
algo.weightUpdate(W)
predictions = algo.fit(trainset).test(testset)
# predictions = algo.fit(trainset)




PredictM = np.zeros(s)
for it in predictions:
    PredictM[int(it[0]),int(it[1])] = it[3]
print (PredictM)
PM = pd.DataFrame(PredictM)
PM.to_csv("PredictionMatrix.csv")
# print (trainset.all_items())

NeighborM = np.zeros([trainset.n_items , K])
for i,item in enumerate(trainset.all_items()):
    NeighborM[i] = algo.get_neighbors(item,k = K)
NM = pd.DataFrame(NeighborM)
NM.to_csv("NeighborMatrix.csv")
    # print ("Inner id: ",iterator_neighbors[1],"Raw id:")
    # print (algo.trainset.to_raw_uid(1))
    
# print Neighbor



# u_id:user id      i_id:item id
# un_: user number  in_: item number
# real rating matrix: r[u_id][i_id]
# estimate : r_h[m][u_id][i_id]
# ###########similarity matrix: (item_based) s[i_id][i_id]
# rating interval: 
# weight matrix: W[u_id][i_id]
# k_neighbor: k_neighbor_li


# calculate denominator

'''
w_err_sum = 0
# for u in range(un_):
for user,u_id in enumerate(trainset.all_users()):
    for item,i_id in enumerate(trainset.all_items()):
        for k_id in Neighbor[item]: #j neighbor
            w_err_sum += W[u_id][k_id]*abs_err

# calculate errRate
errRate = 0
for u in range(un_):
    for i in range(in_):        
        abs_err = abs(r[u_id][i_id]-r_h[m][u_id][i_id])
        
        w_sum = 0
        for k_id in k_neighbor_li:
            w_sum += W[u_id][k_id]
        errRate += w_sum*abs_err**2/D

errRate = errRate/w_err_sum

#calculate sign of neighbor (u,j)
# SGN matrix: SGN[u_id][i_id][k_id]
for u in range(un_):
    for i in range(in_): 
        for k_id in k_neighbor_li:
            SGN[u_id][i_id][k_id] = np.sign(r[u_id][i_id]-r_h[m][u_id][i_id])*np.sign(r[u_id][k_id]-r_h[m][u_id][i_id])

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
            W[u_id][k_id] = 1 + SGN[u_id][i_id][k_id]*errRate/(1-errRate)*UE[u_id][k_id]*rho

#Normalize
W = W/np.sum(W)*un_*in_

#Prediction
# At the very beginning, set:
Pred = 0

# recommender weight : recm_w
recm_w = np.log(1/errRate-1)

Pred += r_h[m]*recm_w
'''