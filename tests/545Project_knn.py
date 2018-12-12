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



'''
PredictM = np.ones([trainset.n_users, trainset.n_items])
for it in predictions:
    PredictM[it[0],it[1]] = it[3]
print (PredictM)
# print (trainset.all_items())
''
Neighbor = np.zeros([trainset.n_items , K])
for i,item in enumerate(trainset.all_items()):
    Neighbor[i] = algo.get_neighbors(item,k = K)
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

#########################################################
#                      Formula (7)                      #
#########################################################

# 1. calculate denominator (公式7括号里乘号左边的分母，为一个定值，括号外的sum对其不产生影响)

w_err_sum = 0
# for u in range(un_):
for user,u_id in enumerate(trainset.all_users()):
    for item,i_id in enumerate(trainset.all_items()):
        abs_err = abs(r[u_id][i_id]-r_h[m][u_id][i_id])   #|r_ui-r_ui^m|,r_h表示r_hat预测值,m表示第m个recommender
        for k_id in Neighbor[item]:                       #j neighbor
            w_err_sum += W[u_id][k_id]*abs_err            #单独计算的左边分式的分母

# 2. calculate errRate -- err(R_m)

errRate = 0
for u in range(un_):
    for i in range(in_):
        abs_err = abs(r[u_id][i_id]-r_h[m][u_id][i_id])   #|r_ui-r_ui^m|

        w_sum = 0
        for k_id in k_neighbor_li:
            w_sum += W[u_id][k_id]
        errRate += w_sum*abs_err**2/D                     #括号外的sum计算(除去上面的分母部分)

errRate = errRate/w_err_sum                               #括号外的sum计算/单独计算的分母


########################################################
#                      Formula(8)                      #
########################################################
# calculate sign of neighbor (u,j)
# SGN matrix: SGN[u_id][i_id][k_id]

for u in range(un_):                                      # un_: user number
    for i in range(in_):                                  # in_: item number
        for k_id in k_neighbor_li:                        # k_id: j neighbor
            SGN[u_id][i_id][k_id] = np.sign(r[u_id][i_id]-r_h[m][u_id][i_id])*np.sign(r[u_id][k_id]-r_h[m][u_id][i_id])
                                            #sgn(r_ui-r_ui^m)                *        sgn(r_uj-r_ui^m)


########################################################
#                      Formula(9)                      #
########################################################
# UE[u_id][i_id]
# m = current recommender

yita = 0.5
for u in range(un_):                                      # un_: user number
    for i in range(in_):                                  # in_: item number
        UE[u_id][i_id] = 1 + yita*r[u_id][i_id]/D         # r_ui不受m影响，m次iteration相加后/m仍为r_ui
        for mm in range(m):
            UE[u_id][i_id] -= yita*r_h[mm][u_id][i_id]/m/D
                                                          # 每循环一次mm，减去第mm个recommender中的预测值*yita/m/D


########################################################
#                     Formula(10)                      #
########################################################
# Update weights
# rho belongs to [0.2,0.3,0.4,0.5,0.6]

for u in range(un_):                                      # un_: user number
    for i in range(in_):                                  # in_: item number
        for k_id in k_neighbor_li:                        # k_id: j neighbor
            W[u_id][k_id] = 1 + SGN[u_id][i_id][k_id]*errRate/(1-errRate)*UE[u_id][k_id]*rho
                                                          # same as Formula(10), w_uj^1恒等于1


########################################################
#                Algorithm for-loop(4)                 #
########################################################
#Normalize
W = W/np.sum(W)*un_*in_                                  # un_: user number  in_: item number

########################################################
#                 Algorithm Prediction                 #
########################################################
#Prediction
# At the very beginning, set:
Pred = 0

# recommender weight : recm_w
recm_w = np.log(1/errRate-1)

Pred += r_h[m]*recm_w
'''
