'''The validation module contains the cross_validate function, inspired from
the mighty scikit learn.'''

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import time

import numpy as np
from joblib import Parallel
from joblib import delayed
from six import iteritems

from .split import get_cv
from .. import accuracy
from surprise import PredictionImpossible
from surprise import Prediction
import pandas as pd

def cross_validate_baseline(algo, data, measures=['rmse', 'mae'], cv=None,
                   return_train_measures=False, n_jobs=1,
                   pre_dispatch='2*n_jobs', verbose=False):
    '''
    Run a cross validation procedure for a given algorithm, reporting accuracy
    measures and computation times.

    See an example in the :ref:`User Guide <cross_validate_example>`.

    Args:
        algo(:obj:`AlgoBase \
            <surprise.prediction_algorithms.algo_base.AlgoBase>`):
            The algorithm to evaluate.
        data(:obj:`Dataset <surprise.dataset.Dataset>`): The dataset on which
            to evaluate the algorithm.
        measures(list of string): The performance measures to compute. Allowed
            names are function names as defined in the :mod:`accuracy
            <surprise.accuracy>` module. Default is ``['rmse', 'mae']``.
        cv(cross-validation iterator, int or ``None``): Determines how the
            ``data`` parameter will be split (i.e. how trainsets and testsets
            will be defined). If an int is passed, :class:`KFold
            <surprise.model_selection.split.KFold>` is used with the
            appropriate ``n_splits`` parameter. If ``None``, :class:`KFold
            <surprise.model_selection.split.KFold>` is used with
            ``n_splits=5``.
        return_train_measures(bool): Whether to compute performance measures on
            the trainsets. Default is ``False``.
        n_jobs(int): The maximum number of folds evaluated in parallel.

            - If ``-1``, all CPUs are used.
            - If ``1`` is given, no parallel computing code is used at all,\
                which is useful for debugging.
            - For ``n_jobs`` below ``-1``, ``(n_cpus + n_jobs + 1)`` are\
                used.  For example, with ``n_jobs = -2`` all CPUs but one are\
                used.

            Default is ``1``.
        pre_dispatch(int or string): Controls the number of jobs that get
            dispatched during parallel execution. Reducing this number can be
            useful to avoid an explosion of memory consumption when more jobs
            get dispatched than CPUs can process. This parameter can be:

            - ``None``, in which case all the jobs are immediately created\
                and spawned. Use this for lightweight and fast-running\
                jobs, to avoid delays due to on-demand spawning of the\
                jobs.
            - An int, giving the exact number of total jobs that are\
                spawned.
            - A string, giving an expression as a function of ``n_jobs``,\
                as in ``'2*n_jobs'``.

            Default is ``'2*n_jobs'``.
        verbose(int): If ``True`` accuracy measures for each split are printed,
            as well as train and test times. Averages and standard deviations
            over all splits are also reported. Default is ``False``: nothing is
            printed.

    Returns:
        dict: A dict with the following keys:

            - ``'test_*'`` where ``*`` corresponds to a lower-case accuracy
              measure, e.g. ``'test_rmse'``: numpy array with accuracy values
              for each testset.

            - ``'train_*'`` where ``*`` corresponds to a lower-case accuracy
              measure, e.g. ``'train_rmse'``: numpy array with accuracy values
              for each trainset. Only available if ``return_train_measures`` is
              ``True``.

            - ``'fit_time'``: numpy array with the training time in seconds for
              each split.

            - ``'test_time'``: numpy array with the testing time in seconds for
              each split.

    '''

    measures = [m.lower() for m in measures]

    cv = get_cv(cv)

    delayed_list = (delayed(fit_and_score)(algo, trainset, testset, measures,
                                           return_train_measures)
                    for (trainset, testset) in cv.split(data))
    out = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch)(delayed_list)

    (test_measures_dicts,
     train_measures_dicts,
     fit_times,
     test_times) = zip(*out)

    test_measures = dict()
    train_measures = dict()
    ret = dict()
    for m in measures:
        # transform list of dicts into dict of lists
        # Same as in GridSearchCV.fit()
        test_measures[m] = np.asarray([d[m] for d in test_measures_dicts])
        ret['test_' + m] = test_measures[m]
        if return_train_measures:
            train_measures[m] = np.asarray([d[m] for d in
                                            train_measures_dicts])
            ret['train_' + m] = train_measures[m]

    ret['fit_time'] = fit_times
    ret['test_time'] = test_times

    if verbose:
        print_summary(algo, measures, test_measures, train_measures, fit_times,
                      test_times, cv.n_splits)

    return ret


def fit_and_score(algo, trainset, testset, measures,
                  return_train_measures=False):
    '''Helper method that trains an algorithm and compute accuracy measures on
    a testset. Also report train and test times.

    Args:
        algo(:obj:`AlgoBase \
            <surprise.prediction_algorithms.algo_base.AlgoBase>`):
            The algorithm to use.
        trainset(:obj:`Trainset <surprise.trainset.Trainset>`): The trainset.
        trainset(:obj:`testset`): The testset.
        measures(list of string): The performance measures to compute. Allowed
            names are function names as defined in the :mod:`accuracy
            <surprise.accuracy>` module.
        return_train_measures(bool): Whether to compute performance measures on
            the trainset. Default is ``False``.

    Returns:
        tuple: A tuple containing:

            - A dictionary mapping each accuracy metric to its value on the
            testset (keys are lower case).

            - A dictionary mapping each accuracy metric to its value on the
            trainset (keys are lower case). This dict is empty if
            return_train_measures is False.

            - The fit time in seconds.

            - The testing time in seconds.
    '''


















    start_fit = time.time()
    algo.fit(trainset)
    fit_time = time.time() - start_fit
    start_test = time.time()
    predictions = algo.test(testset)
    test_time = time.time() - start_test

    if return_train_measures:
        train_predictions = algo.test(trainset.build_testset())

    test_measures = dict()
    train_measures = dict()
    for m in measures:
        f = getattr(accuracy, m.lower())
        test_measures[m] = f(predictions, verbose=0)
        if return_train_measures:
            train_measures[m] = f(train_predictions, verbose=0)

    return test_measures, train_measures, fit_time, test_time


def print_summary(algo, measures, test_measures, train_measures, fit_times,
                  test_times, n_splits):
    '''Helper for printing the result of cross_validate.'''

    print('Evaluating {0} of algorithm {1} on {2} split(s).'.format(
          ', '.join((m.upper() for m in measures)),
          algo.__class__.__name__, n_splits))
    print()

    row_format = '{:<18}' + '{:<8}' * (n_splits + 2)
    s = row_format.format(
        '',
        *['Fold {0}'.format(i + 1) for i in range(n_splits)] + ['Mean'] +
        ['Std'])
    s += '\n'
    s += '\n'.join(row_format.format(
        key.upper() + ' (testset)',
        *['{:1.4f}'.format(v) for v in vals] +
        ['{:1.4f}'.format(np.mean(vals))] +
        ['{:1.4f}'.format(np.std(vals))])
        for (key, vals) in iteritems(test_measures))
    if train_measures:
        s += '\n'
        s += '\n'.join(row_format.format(
            key.upper() + ' (trainset)',
            *['{:1.4f}'.format(v) for v in vals] +
            ['{:1.4f}'.format(np.mean(vals))] +
            ['{:1.4f}'.format(np.std(vals))])
            for (key, vals) in iteritems(train_measures))
    s += '\n'
    s += row_format.format('Fit time',
                           *['{:.2f}'.format(t) for t in fit_times] +
                           ['{:.2f}'.format(np.mean(fit_times))] +
                           ['{:.2f}'.format(np.std(fit_times))])
    s += '\n'
    s += row_format.format('Test time',
                           *['{:.2f}'.format(t) for t in test_times] +
                           ['{:.2f}'.format(np.mean(test_times))] +
                           ['{:.2f}'.format(np.std(test_times))])
    print(s)



def adaboost(algo, trainset, testset):


    m = 10 # Number of Adaboost iterations
    D = 5-1 # Rating range
    yita = 0.5 # yita denotes how much the average sample error influences the update process, set 0.5 by experience
    rho = 0.7 # Adaboost update rate, rho falls within [0.2,0.3,0.4,0.5,0.6]
    recm_w = np.ones(m) # Adaboost weight

    # Data declaration and spliting into train & test
    data = Dataset.load_builtin('ml-100k')
    # data = Dataset.load_builtin('ml-1m')
    WholeSet = data.build_full_trainset()  # Total data set for universal indexing
    trainset, ABtestset = train_test_split(data, test_size=0.5) #Split data using test_size

    # choosing algorithm: ItemBased / UserBased

    # sim_options = {'name': 'pearson_baseline', 'user_based': False}
    bsl_options = {'method': 'sgd','reg': 1.2}
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
    accuracy.rmse(predictions)
    print("individual rmse:", ABRMSE)





############################################     Prediction Model    #########################################
def predict(uid, iid, r_ui=None, clip=True, verbose=False,WholeSet,ABPredictM,trainset):
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

