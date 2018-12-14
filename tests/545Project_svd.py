# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 22:09:32 2018

@author: Zhaoer li, Huiwen Cao, Shidong Zhang, Weiqi Zhou
"""

import numpy as np
import surprise
from surprise import BaselineOnly
from surprise import Dataset
from surprise import get_dataset_dir
from surprise.model_selection import train_test_split
import pandas as pd
from surprise import accuracy
from surprise import PredictionImpossible
from surprise import Prediction
from operator import itemgetter
from surprise import AlgoBase
from surprise import NMF
from surprise import SVD

#Parameter Declaration

m = 5 # Number of Adaboost iterations
D = 5-1 # Rating range
yita = 0.5 # yita denotes how much the average sample error influences the update process, set 0.5 by experience
rho = 0.2 # Adaboost update rate, rho falls within [0.2,0.3,0.4,0.5,0.6]
recm_w = np.ones(m) # Adaboost weight

# Data declaration and spliting into train & test
data = Dataset.load_builtin('ml-100k')
# data = Dataset.load_builtin('ml-1m')
WholeSet = data.build_full_trainset()  # Total data set for universal indexing
trainset, ABtestset = train_test_split(data, test_size=0.20) #Split data using test_size

# choosing algorithm: ItemBased / UserBased

# sim_options = {'name': 'pearson_baseline', 'user_based': False}
sim_options = {'user_based': False}
bsl_options = {'method': 'sgd',
               'learning_rate': .00005}
# Initialize testset T_train for Adaboost iterations, it is identical with trainset
testset = [None]*trainset.n_ratings
iter = 0
for uid,iid,ratings in trainset.all_ratings():
    # print("is uid,iid int or not?", isinstance(uid, int))
    ruid = trainset.to_raw_uid(uid)
    riid = trainset.to_raw_iid(iid)
    # print("and raw ids are:",ruid,riid)
    testset[iter] = [ruid,riid,ratings]
    # print("testset element are:", testset[iter])
    iter+=1
# Output testset to a csv file
PM = pd.DataFrame(testset)
PM.to_csv("TestSet.csv")


# Initializing algorithm with predefined options
# algo = NMF(biased = True)
algo = SVD(biased = True)
# algo = KNNBaseline()

# Initializing sizes for Adaboost parameter matrices
size_ui = (trainset.n_users + 1, trainset.n_items + 1)
size_mui = (m,trainset.n_users + 1, trainset.n_items + 1)
size_wmui = (m,WholeSet.n_users + 1, WholeSet.n_items + 1)

# Initializing weight matrix
W = np.ones(size_ui)
# Initializing Adaboost Prediction matrix from ABtestset
ABPredictM = np.zeros(size_wmui)
# Initializing weight-update Prediction matrix from T_train
PredictM = np.zeros(size_mui)
# Initializing RMSE vector to store RMSE of ABtestset from each model in Adaboost iteration
ABRMSE = np.zeros(m,dtype=float)

# Initializing Rating Matrix to store true ratings from T_train
RatingM = np.zeros(size_ui)
for uid, iid, rating in trainset.all_ratings():
    RatingM[uid,iid] = rating

# Starting the main Adaboost loop
for mm in range(m):

                                                            #Obtain prediction using current W
    ############################################    Adaboost Step 1   #########################################
    # algo = BaselineOnly(bsl_options = bsl_options)
    algo.weightUpdate(W)
    predictions = algo.fit(trainset).test(ABtestset)
    # predictions = algo.test(ABtestset)
    ABRMSE[mm] = accuracy.rmse(predictions)
    for (ruid, riid, _, est, _,_) in predictions:
        # print("predictM loop: ", ruid,riid,est)
        uid = WholeSet.to_inner_uid(ruid)
        iid = WholeSet.to_inner_iid(riid)
        ABPredictM[mm, uid, iid] = est

    ############################################    Adaboost Step 2   #########################################
    # predictions = algo.fit(trainset).test(testset)
    # algo = BaselineOnly(bsl_options = bsl_options)
    # algo.weightUpdate(W)
    predictions = algo.fit(trainset).test(testset)
    # predictions = algo.test(testset)
    # PM = pd.DataFrame(predictions)
    # PM.to_csv("CurrentPredictions.csv")
    # accuracy.rmse(predictions)                                                      #print current RMSE accuracy

    # sortedlist = sorted(predictions, key=lambda tup: tup[5], reverse=True)[:T]      #Sort prediction in descending order of rating errors for the fist T element

    # print("trainset size:", trainset.n_users, trainset.n_items)
    # print("trainset iid:", trainset.to_inner_iid('1080'))
    # print("wholeset iid:", WholeSet.to_inner_iid('1080'))
    for (ruid, riid, _, est, _,_) in predictions:                                   #Update current weight-update Prediction matrix
        # print("predictM loop: ", ruid,riid,est)
        uid = trainset.to_inner_uid(ruid)
        iid = trainset.to_inner_iid(riid)
        PredictM[mm][uid][iid] = est

    UE = np.ones(size_ui)                                                           #Initializing Adaboost parameters
    SGNM = (trainset.n_users+1)*[(trainset.n_items+1)*[None]]
    errRate = 0
    w_err_sum = 0

############################################    Adaboost Iteration loop#########################################
    for ruid, riid, rating in testset:
        # from raw ids to inner ids
        uid = trainset.to_inner_uid(ruid)
        iid = trainset.to_inner_iid(riid)
        #########################################################
        #                      Formula (7)                      #
        #########################################################
        abs_err = abs(rating - PredictM[mm][uid][iid])
        ########################################################
        #                      Formula(9)                      #
        ########################################################
        UE[uid][iid] = 1 + yita * rating
        for mmm in range(mm+1):
            ########################################################
            #                      Formula(9)                      #
            ########################################################
            UE[uid][iid] -= yita * PredictM[mmm][uid][iid] / (mm+1) / D

        ########################################################
        #                      Formula(7)                      #
        ########################################################
        w_err_sum += W[uid][iid]
        ########################################################
        #                      Formula(7)                      #
        ########################################################
        errRate += W[uid][iid]*(abs_err)/D
    ########################################################
    #                      Formula(7)                      #
    ########################################################
    errRate = errRate / w_err_sum

    recm_w[mm] = np.log((1 - errRate) / errRate )                                # Calculating Adaboost Prediction Model weights
    PM = pd.DataFrame(UE)
    PM.to_csv("UEMatrix.csv")
    # for uid, iid, rating in trainset.all_ratings():
    ############################################    Adaboost Step 3   #########################################
    if(mm<m-1):
        for (ruid, riid, rating, _, _, _) in predictions:
            # print("predictM loop: ", ruid,riid,est)
            uid = trainset.to_inner_uid(ruid)
            iid = trainset.to_inner_iid(riid)
            W[uid][iid] = (1 - (errRate / (1-errRate)) * UE[uid][iid] * rho)    # Update Weights matrix
            # print("updated W paras:errRate,UE,rho:",errRate,UE[uid][iid],rho  )
            # print("updated W element:", W[uid][iid])
    ############################################    Adaboost Step 4   #########################################
    W = W / np.sum(W) * trainset.n_users * trainset.n_items
    PM = pd.DataFrame(W)
    PM.to_csv("WeightMatrix.csv")
    # print("w_err_sum is:",w_err_sum)
    # print("errRate is:",errRate)

# print("boostweight: ",recm_w)
############################################    Normalize Prediction Model Weights   #########################################
recm_w = recm_w / sum(recm_w)






############################################     Prediction Model    #########################################
def predict(uid, iid, r_ui=None, clip=True, verbose=False):
    """Compute the rating prediction for given user and item.

    The ``predict`` method converts raw ids to inner ids and then calls the
    ``estimate`` method which is defined in every derived class. If the
    prediction is impossible (e.g. because the user and/or the item is
    unkown), the prediction is set according to :meth:`default_prediction()
    <surprise.prediction_algorithms.algo_base.AlgoBase.default_prediction>`.

    Args:
        uid: (Raw) id of the user. See :ref:`this note<raw_inner_note>`.
        iid: (Raw) id of the item. See :ref:`this note<raw_inner_note>`.
        r_ui(float): The true rating :math:`r_{ui}`. Optional, default is
            ``None``.
        clip(bool): Whether to clip the estimation into the rating scale,
            that was set during dataset creation. For example, if
            :math:`\\hat{r}_{ui}` is :math:`5.5` while the rating scale is
            :math:`[1, 5]`, then :math:`\\hat{r}_{ui}` is set to :math:`5`.
            Same goes if :math:`\\hat{r}_{ui} < 1`.  Default is ``True``.
        verbose(bool): Whether to print details of the prediction.  Default
            is False.

    Returns:
        A :obj:`Prediction\
        <surprise.prediction_algorithms.predictions.Prediction>` object
        containing:

        - The (raw) user id ``uid``.
        - The (raw) item id ``iid``.
        - The true rating ``r_ui`` (:math:`\\hat{r}_{ui}`).
        - The estimated ratino
ig (:math:`\\hat{r}_{ui}`).
        - Some additional details about the prediction that might be useful
          for later analysis.
    """

    # Convert raw ids to inner ids
    # print("inner ids: ", uid, ", ", iid)
    try:

        iuid = WholeSet.to_inner_uid(uid)
        # print('uid = ',uid,'iuid = ', iuid)
    except ValueError:
        print("545: uid error!")
        iuid = 'UKN__' + str(uid)
    try:
        iiid = WholeSet.to_inner_iid(iid)
    except ValueError:
        print("545: iid error!")
        iiid = 'UKN__' + str(iid)

    details = {}
    try:
        est = 0.0
        for mm in range(m):
############################################    Estimation from Adaboost Prediction Model #########################################
            est += ABPredictM[mm][iuid][iiid] * recm_w[mm]
        # If the details dict was also returned
        if isinstance(est, tuple):
            est, details = est
        details['was_impossible'] = False

    except PredictionImpossible as e:
        est = default_prediction()
        details['was_impossible'] = True
        details['reason'] = str(e)

    # clip estimate into [lower_bound, higher_bound]
    if clip:
        lower_bound, higher_bound = trainset.rating_scale
        est = min(higher_bound, est)
        est = max(lower_bound, est)

    pred = Prediction(uid, iid, r_ui, est, details,abs(r_ui - est))

    if verbose:
        print(pred)

    return pred

############################################    Calling Prediction Model Weights   #########################################
predictions = [predict(uid,iid, r_ui_trans,verbose=False) for (uid, iid, r_ui_trans) in ABtestset]

############################################    Printing RMSE of Adaboost Prediction Model and individual RMSE from each Adaboost iteration  #########################################
accuracy.rmse(predictions)
print("individual rmse:",ABRMSE)




#########################################################
#                      Formula (7)                      #
#########################################################

# 1. calculate denominator (公式7括号里乘号左边的分母，为一个定值，括号外的sum对其不产生影响)


# for u in range(un_):


    # print("test all_ratings()")
    # print(uid,iid,rating)
    # if w_err_sum > 5:
    #     break
    # else:
    #     w_err_sum += 1

# for user,u_id in enumerate(trainset.all_users()):
#     for item,i_id in enumerate(trainset.all_items()):
#         abs_err = abs(r[u_id][i_id]-r_h[m][u_id][i_id])   #|r_ui-r_ui^m|,r_h表示r_hat预测值,m表示第m个recommender
#         for k_id in Neighbor[item]:                       #j neighbor
#             w_err_sum += W[u_id][k_id]*abs_err            #单独计算的左边分式的分母





# 2. calculate errRate -- err(R_m)

# errRate = 0
# for u in range(un_):
#     for i in range(in_):
#         abs_err = abs(r[u_id][i_id]-r_h[m][u_id][i_id])   #|r_ui-r_ui^m|
#
#         w_sum = 0
#         for k_id in k_neighbor_li:
#             w_sum += W[u_id][k_id]
#         errRate += w_sum*abs_err**2/D                     #括号外的sum计算(除去上面的分母部分)
#
# errRate = errRate/w_err_sum                               #括号外的sum计算/单独计算的分母


########################################################
#                      Formula(8)                      #
########################################################
# calculate sign of neighbor (u,j)
# SGN matrix: SGN[u_id][i_id][k_id]
# SGNM = np.zeros((trainset.n_users,trainset.n_items,K))
# for u in range(un_):                                      # un_: user number
#     for i in range(in_):                                  # in_: item number
#         for k_id in k_neighbor_li:                        # k_id: j neighbor
#             SGN[u_id][i_id][k_id] = np.sign(r[u_id][i_id]-r_h[m][u_id][i_id])*np.sign(r[u_id][k_id]-r_h[m][u_id][i_id])
#                                             #sgn(r_ui-r_ui^m)                *        sgn(r_uj-r_ui^m)
# for uid, iid, rating in trainset.all_ratings():
#     for kid in NeighborM[iid]:
#         SGNM[uid][iid][kid] = np.sign(rating - PredictM[uid][iid]) * np.sign(RatingM[uid,kid]- PredictM[uid][iid])


########################################################
#                      Formula(9)                      #
########################################################
# UE[u_id][i_id]
# m = current recommender

# yita = 0.5
# for u in range(un_):                                      # un_: user number
#     for i in range(in_):                                  # in_: item number
#         UE[u_id][i_id] = 1 + yita*r[u_id][i_id]/D         # r_ui不受m影响，m次iteration相加后/m仍为r_ui
#         for mm in range(m):
#             UE[u_id][i_id] -= yita*r_h[mm][u_id][i_id]/m/D

                                                         # 每循环一次mm，减去第mm个recommender中的预测值*yita/m/D




########################################################
#                     Formula(10)                      #
########################################################
# Update weights
# rho belongs to [0.2,0.3,0.4,0.5,0.6]

# for u in range(un_):                                      # un_: user number
#     for i in range(in_):                                  # in_: item number
#         for k_id in k_neighbor_li:                        # k_id: j neighbor
#             W[u_id][k_id] = 1 + SGN[u_id][i_id][k_id]*errRate/(1-errRate)*UE[u_id][k_id]*rho
#                                                           # same as Formula(10), w_uj^1恒等于1
#
#
# ########################################################
# #                Algorithm for-loop(4)                 #
# ########################################################
# #Normalize
# W = W/np.sum(W)*un_*in_                                  # un_: user number  in_: item number
#
# ########################################################
# #                 Algorithm Prediction                 #
# ########################################################
# #Prediction
# # At the very beginning, set:
# Pred = 0
#
# # recommender weight : recm_w
# recm_w = np.log(1/errRate-1)
#
# Pred += r_h[m]*recm_w
