# -*- coding: utf-8 -*-
"""
Created on Tue May 24 11:29:26 2022

@author: admin
"""

import pandas as pd
import numpy as np 
from sklearn import svm
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

def read_seqs(path):
    seqs = pd.read_csv(path, header=None)
    seqs = seqs[seqs[0].str.isalpha()][0]
    return seqs

def read_activity(path):
    activity = pd.read_csv(path, header=None)
    activity = activity[1]
    return activity.values

def one_hot_feature(seqs):
    peptide_index = np.eye(20)
    peptide_one_hot_matric = np.zeros((len(seqs), 20*3))
    row = 0
    for seq in seqs:
        col = 0
        for peptide in seq:
            peptide_one_hot_matric[row][col*20:col*20+20] = peptide_index[amino_acid.index(peptide)]
            col += 1
        row += 1
    #dipeptide_feature
    dipeptide_one_hot_matric = np.zeros((len(seqs), 400))
    dipeptide_list = []
    for peptide in amino_acid:        
        piece = map(lambda pep: pep + peptide, amino_acid)
        dipeptide_list += piece
    row = 0
    for seq in seqs:
        for index in range(len(seq)-1):
            dipeptide = seq[0+index:2+index]
            dipeptide_one_hot_matric[row][dipeptide_list.index(dipeptide)] += 1
        row += 1
    one_hot_matric = np.c_[peptide_one_hot_matric, dipeptide_one_hot_matric]    
    return one_hot_matric

def AAC(seqs):
    AAC_matric = np.zeros((len(seqs), 20))
    row = 0
    for seq in seqs:
        for peptide in seq:
            AAC_matric[row][amino_acid.index(peptide)] += 1/3
        row += 1
    return AAC_matric
    
def SVM_opt(X, y, length):
    y_mean = np.full((1,length), np.sum(y)/length)
    count = -1
    global verify
    verify = np.zeros((576,4))
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    for first_iteration in range(-1,7):
        c = 2**first_iteration
        for second_iteration in range(-8,0):
            p = 2**second_iteration
            for third_iteration in range(-8,1):
                g = 2**third_iteration 
                YPred = np.zeros(length)        
                count = count + 1
                for train_index, test_index in loo.split(X):      
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]             
                    regr = svm.SVR(kernel ='rbf',degree = 3,gamma = g , coef0 = 0.0,
    		    tol = 0.001, C = c, epsilon = p, shrinking = True, cache_size = 40,		
    		    verbose = False,max_iter = -1)
                    regr = regr.fit(X_train, y_train)
                    y_pre = regr.predict(X_test)
                    YPred[test_index] = y_pre
                global num_opt
                num_opt = (1-np.sum((y-YPred)**2)/np.sum((y-y_mean)**2))
                verify[count][0] = c
                verify[count][1] = p
                verify[count][2] = g
                verify[count][3] = num_opt
    opt = verify[np.argsort(verify[:,3])]
    g_opt, c_opt, p_opt = opt[-1,2], opt[-1,0], opt[-1,1]
    regr = svm.SVR(kernel ='rbf',degree = 3,gamma = g_opt , coef0 = 0.0,
    		    tol = 0.001, C = c_opt, epsilon = p_opt, shrinking = True, cache_size = 40,		
    		    verbose = False,max_iter = -1)
    SVM_reg = regr.fit(X, y)
    Y_predication = SVM_reg.predict(X)
    return Y_predication, opt[-1,3], g_opt, c_opt, p_opt

# random_forest
def RF(X, y, length):
    Q2_array = np.zeros((101,2))
    for n in range(1,101):
        clf = RandomForestRegressor(n_estimators = n, random_state = 0)
        y_mean = np.full((1,length), np.sum(y)/length)
        loo = LeaveOneOut()
        loo.get_n_splits(X)
        YPred = np.zeros(length)
        for train_index, test_index in loo.split(X):      
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = clf.fit(X_train, y_train)
            y_pre = clf.predict(X_test)
            YPred[test_index] = y_pre
        Q2 = (1-np.sum((y - YPred)**2)/np.sum((y-y_mean)**2))
        Q2_array[n][0] = Q2
        Q2_array[n][1] = n
    opt = Q2_array[np.argsort(Q2_array[:,0])]
    n_forest = opt[-1, 1]
    clf = RandomForestRegressor(n_estimators = int(n_forest), random_state = 0)
    clf = clf.fit(X,y)
    Y_predication = clf.predict(X)
    return Y_predication, opt[-1,0], int(n_forest)


# GradientBoostingRegressor
def GBR(X, y, length):
    Q2_array = np.zeros((1001, 2))
    for n in range(1, 1001):
        regr = GradientBoostingRegressor(random_state=0, n_estimators=n)
        y_mean = np.full((1,length), np.sum(y)/length)
        loo = LeaveOneOut()
        loo.get_n_splits(X)
        YPred = np.zeros(length) 
        for train_index, test_index in loo.split(X):      
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index] 
            AdB_reg = regr.fit(X_train, y_train)
            y_pre = AdB_reg.predict(X_test)
            YPred[test_index] = y_pre
        Q2 = (1-np.sum((y - YPred)**2)/np.sum((y-y_mean)**2))
        Q2_array[n][0] = Q2
        Q2_array[n][1] = n    
    opt = Q2_array[np.argsort(Q2_array[:,0])]
    n_estimator = opt[-1, 1]
    clf = GradientBoostingRegressor(random_state=0, n_estimators = int(n_estimator))
    clf.fit(X, y)
    Y_predication = clf.predict(X)
    return Y_predication, opt[-1, 0], int(n_estimator)

# MLP
def MLP(X, y, length):
    regr = MLPRegressor(random_state=1, max_iter=200)
    y_mean = np.full((1,length), np.sum(y)/length)
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    YPred = np.zeros(length) 
    for train_index, test_index in loo.split(X):      
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index] 
        AdB_reg = regr.fit(X_train, y_train)
        y_pre = AdB_reg.predict(X_test)
        YPred[test_index] = y_pre
    Q2 = (1-np.sum((y - YPred)**2)/np.sum((y-y_mean)**2))
    regr.fit(X, y)
    Y_prediction = regr.predict(X)
    return Y_prediction, Q2, YPred


# indicators
def Indicators(y_hat, y, length):
    MAE = np.mean(np.abs(y - y_hat))
    MSE = np.mean(np.square(y - y_hat))
    RMSE = np.sqrt(np.mean(np.square(y - y_hat)))
    MAPE = np.mean(np.abs((y - y_hat) / y)) * 100
    y_mean = np.full((1, length), np.sum(y)/ length)
    R2 = (1-np.sum((y - y_hat)**2)/np.sum((y-y_mean)**2))
    return MAE, MSE, RMSE, MAPE, R2

# get data
amino_acid = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
place = [1, 2, 3, 4, 5, 6, 'data']
i = place[5]
path = r'C:\Users\li\Desktop\modification\modi_method\fusion_{}.txt'.format(i)
seqs = read_seqs(path)
activity = read_activity(path)

# feature transform
one_hot_feature = one_hot_feature(seqs)
AAC_matric = AAC(seqs)
whole_feature = np.c_[one_hot_feature, AAC_matric]
length = len(seqs)


SVM_opt_result, SVM_Q2, g_opt, c_opt, p_opt = SVM_opt(whole_feature, activity, length)
SVM_indicators = Indicators(SVM_opt_result, activity, length)

RF_result, RF_Q2, RF_n = RF(whole_feature, activity, length)
RF_indicators = Indicators(RF_result, activity, length)


GBR_result, GBR_Q2, GBR_n = GBR(whole_feature, activity, length)
GBR_indicators = Indicators(GBR_result, activity, length)

MLP_result, MLP_Q2, MLP_YPred = MLP(whole_feature, activity, length)
MLP_indicators = Indicators(MLP_result, activity, length)

