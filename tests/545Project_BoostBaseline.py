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
from surprise.model_selection.split import get_cv
#Parameter Declaration

############################################     Prediction Model    #########################################
def predict(uid, iid, r_ui=None, clip=True, verbose=False):
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



m = 4 # Number of Adaboost iterations
D = 5-1 # Rating range
yita = 0.5 # yita denotes how much the average sample error influences the update process, set 0.5 by experience
rho = 0.7 # Adaboost update rate, rho falls within [0.2,0.3,0.4,0.5,0.6]
recm_w = np.ones(m) # Adaboost weight

# Data declaration and spliting into train & test
data = Dataset.load_builtin('ml-100k')
# data = Dataset.load_builtin('ml-1m')
WholeSet = data.build_full_trainset()  # Total data set for universal indexing
# trainset, ABtestset = train_test_split(data, test_size=0.5) #Split data using test_size
bsl_options = {'method': 'sgd','reg': 1.2}


cv = get_cv(None)
CrossVRMSE = np.zeros(5,dtype=float)
Crossiter = 0
for (trainset, ABtestset) in cv.split(data):
    # choosing algorithm: ItemBased / UserBased

    # sim_options = {'name': 'pearson_baseline', 'user_based': False}

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
    algo = BaselineOnly(bsl_options = bsl_options)
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
        errRate = 0
        w_err_sum = 0

    ############################################    Adaboost Iteration loop#########################################
        for ruid, riid, rating in testset:
            # from raw ids to inner ids
            uid = trainset.to_inner_uid(ruid)
            iid = trainset.to_inner_iid(riid)
            #########################################################
            #                      Formula (11)                      #
            #########################################################
            abs_err = abs(rating - PredictM[mm][uid][iid])
            ########################################################
            #                      Formula(13)                      #
            ########################################################
            UE[uid][iid] = 1 + yita * rating
            for mmm in range(mm+1):
                ########################################################
                #                      Formula(13)                      #
                ########################################################
                UE[uid][iid] -= yita * PredictM[mmm][uid][iid] / (mm+1) / D

            ########################################################
            #                      Formula(11)                      #
            ########################################################
            w_err_sum += W[uid][iid]
            ########################################################
            #                      Formula(11)                      #
            ########################################################
            errRate += W[uid][iid]*(abs_err)/D
        ########################################################
        #                      Formula(11)                      #
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
                ########################################################
                #                      Formula(14)                      #
                ########################################################
                W[uid][iid] = (1 - (errRate / (1-errRate)) * UE[uid][iid] * rho)    # Update Weights matrix
                # print("updated W paras:errRate,UE,rho:",errRate,UE[uid][iid],rho  )
                # print("updated W element:", W[uid][iid])
        ############################################    Adaboost Step 4   #########################################
        # W = W / np.sum(W) * trainset.n_users * trainset.n_items
        PM = pd.DataFrame(W)
        PM.to_csv("WeightMatrix.csv")
        # print("w_err_sum is:",w_err_sum)
        # print("errRate is:",errRate)

    # print("boostweight: ",recm_w)
    ############################################    Normalize Prediction Model Weights   #########################################
    recm_w = recm_w / sum(recm_w)
    ############################################    Calling Prediction Model Weights   #########################################
    predictions = [predict(uid, iid, r_ui_trans, verbose=False) for (uid, iid, r_ui_trans) in ABtestset]

    ############################################    Printing RMSE of Adaboost Prediction Model and individual RMSE from each Adaboost iteration  #########################################
    print("One CF loop Finished, Current RMSE =")
    CrossVRMSE[Crossiter] = accuracy.rmse(predictions)

    Crossiter += 1

print("CrossValidated Mean RMSE", np.mean(CrossVRMSE))










