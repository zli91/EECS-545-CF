from surprise import KNNBaseline
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
import numpy as np
import pandas as pd
from surprise import accuracy
from operator import itemgetter

# Load the movielens-100k dataset (download it if needed).
# data = Dataset.load_builtin('ml-100k')
data = Dataset.load_builtin('ml-1m')
sizefinder = data.build_full_trainset()

# trainset, testset = train_test_split(data, test_size=1.0)
trainset, testset = train_test_split(data, test_size=0.5)
# print(trainset)
# print("trainset iid:",trainset.to_inner_iid('1080'))
# print("wholeset iid:",sizefinder.to_inner_iid('1080'))

MakeTestSet = np.ones((trainset.n_ratings , 3),dtype = int)
iter = 0
# for uid,iid,ratings in trainset.all_ratings():
#     MakeTestSet[iter] = [uid,iid,ratings]
#     iter+=1
# print(MakeTestSet[5][1])
# print(testset[0][1])

# PM = pd.DataFrame(testset)
# PM.to_csv("TestSet.csv")
#
# # Use the famous SVD algorithm.
sim_options = {'name': 'pearson_baseline', 'user_based': False}
algo = KNNBaseline(sim_options=sim_options)
s = (sizefinder.n_users+1,sizefinder.n_items+1)
# trainM = np.zeros(s)
# for uid,iid,rating in sizefinder.all_ratings():
#     trainM[uid][iid] = rating
# PM = pd.DataFrame(trainM)
# PM.to_csv("TrainSet.csv")
#
weight = np.ones(s)
algo.weightUpdate(weight)

# algo.fit(trainset)
# est,details,k_neighbor = [algo.estimate(uid,iid) for (uid, iid, _) in testset[1]]
# NeighborM = (trainset.n_users+1)*[(trainset.n_items+1)*[None]]
# for uid,iid,_ in trainset.all_ratings():
#     _,_,K_neighbor = algo.estimate(uid,iid)
#     NeighborM[uid][iid] = np.array(K_neighbor)[:,0][:10]

# uid = trainset.to_inner_uid(testset[1][0])
# iid = trainset.to_inner_iid(testset[1][1])
# print(len(trainset.ur[uid]))
# print(NeighborM[uid][iid])
# est = algo.estimate(uid,iid)
# print(est,isinstance(est, tuple))

predictions = algo.fit(trainset).test(testset)
#
#
#
#
# accuracy.rmse(predictions)
# print(predictions[3][5])
# sortedlist = sorted(predictions, key=lambda tup: tup[5], reverse = True)[:T]
#
#
# PM = pd.DataFrame(sortedlist)
# PM.to_csv("SortedPredictions.csv")

# print("trainset iid:",trainset.to_inner_iid('1080'))
# print("wholeset iid:",sizefinder.to_inner_iid('1080'))
# size_ui = (sizefinder.n_users + 1, sizefinder.n_items + 1)
#
# PredictM = np.zeros(size_ui)
# for it in predictions:
#     uid = sizefinder.to_inner_uid(it[0])
#     iid = sizefinder.to_inner_iid(it[1])
#     PredictM[uid,iid] = it[3]
# PM = pd.DataFrame(PredictM)
# PM.to_csv("PredictionMatrix.csv")
#
#
#
#
# def predict(uid, iid, r_ui=None):
#     print("inner ids: ", uid, iid, r_ui)
#     return 1
#
# for (uid, iid, r_ui_trans) in MakeTestSet[:5]:
#     print("inner ids: ", uid, iid, r_ui_trans)
    # predict(uid, iid, r_ui_trans)






# predictions = [predict(uid, iid, r_ui_trans) for (uid, iid, r_ui_trans) in testset[:5]]
# print('total number of users: ',trainset.n_users)
# print('testset element',testset[2])
# Run 5-fold cross-validation and print results.
# cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
# print predictions[10]
# print predictions[10][3]