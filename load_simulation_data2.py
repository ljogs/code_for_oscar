# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 16:11:15 2022

@author: LJOGS
"""

import numpy as np

import random

from oscarpackage import oscar,testlossf

from sklearn.linear_model import Lasso, ElasticNet, Ridge


import pandas as pd

import matlab.engine
eng = matlab.engine.start_matlab()

random.seed(10)
seed_pool = random.sample(range(1,1000),100)

OSCAR_MSE_log=[];
Lasso_MSE_log=[];
Elastic_net_MSE_log=[];
OSCAR_df_log=[];
Lasso_df_log=[];
Elastic_net_df_log=[];
clf_MSE_log=[];
clf_df_log=[];

percentage = 0;

for random_seed in seed_pool:
    
    
    percentage=percentage+1;
    
    num_samples = 25
    p=8

    mu = np.zeros(p)

    Temp = np.mat(np.identity(p))


    for i in range(p):
        for j in range(p):
            Temp[i,j]=pow(0.7, abs(i-j))

    rng = np.random.default_rng(random_seed)
    cordataX = rng.multivariate_normal(mu, Temp, size=num_samples) 


    beta=np.array([4,0,0,3,0,0,0,1.5])

    np.random.seed(random_seed)
    cordataY=np.dot(cordataX, beta)+np.random.normal(0,5,num_samples)

    for i in range(p):
        cordataX[:,i] = (cordataX[:,i]-np.mean(cordataX[:,i]))/np.std(cordataX[:,i])
        
    cordataY=(cordataY-np.mean(cordataY))/np.std(cordataY)
    
    std_X = [];
    for i in range(p):
        std_X.append(np.std(cordataX[:,i]))
    
    std_X = np.array(std_X);
    std_Y = np.std(cordataY)
    
    std_beta = std_X*beta/std_Y

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test =  train_test_split(cordataX, cordataY, test_size=0.2, random_state=random_seed)
    

    ####################################OSCAR###############
 
    OSCAR_X_train = pd.DataFrame(X_train);
    OSCAR_X_test = pd.DataFrame(X_test);
    OSCAR_y_train = pd.DataFrame(y_train);
    OSCAR_y_test = pd.DataFrame(y_test);
    
    OSCAR_X_train.to_csv('OSCAR_X_train.csv', index=False, header=False);
    OSCAR_X_test.to_csv('OSCAR_X_test.csv', index=False, header=False);
    OSCAR_y_train.to_csv('OSCAR_y_train.csv', index=False, header=False);
    OSCAR_y_test.to_csv('OSCAR_y_test.csv', index=False, header=False);


    ####################################OSCAR###############
    
    oscar_coef, df= eng.oscar_package(nargout=2)

    oscar_coef=np.array(oscar_coef)
    oscar_coef=oscar_coef[:,0];
    oscar_coef = std_Y*oscar_coef/std_X;
    oscar_mse = np.dot(np.dot(oscar_coef-beta, Temp),(oscar_coef-beta))
    
    OSCAR_MSE_log.append(oscar_mse[0,0])
    OSCAR_df_log.append(df)
    
    
    #########################RIDGE#########################
    
    alphas = np.arange(0.01,1.5,0.1)
    alphas = alphas*p;

    coefs = []
    errors =[]

    for a in alphas:
        clf = Ridge(alpha=a)
        clf.fit(X_train, y_train)
        coefs.append(clf.coef_)
        errors.append(clf.score(X_test, y_test))

    maxerror = np.max(errors)
    maxindex = np.where(errors == maxerror)
    optuning = alphas[maxindex]


    clf.set_params(alpha=optuning[0])
    clf.fit(X_train, y_train)
    clf_coef = clf.coef_
    clf_coef = std_Y*clf_coef/std_X;
    clf_mse = np.dot(np.dot(clf_coef.T-beta.T, Temp),clf_coef-beta)
    
    clf_MSE_log.append(clf_mse[0,0])
    clf_df_log.append(len(set(clf_coef[np.nonzero(clf_coef)])))
    
    
    #############################################LASSO######
    
    alphas = np.arange(0.01,1.3,0.05)
    lasso = Lasso(max_iter=100)
    coefs = []
    errors =[]

    for a in alphas:
        lasso.set_params(alpha=a)
        lasso.fit(X_train, y_train)
        coefs.append(lasso.coef_)
        errors.append(lasso.score(X_test, y_test))

    maxerror = np.max(errors)
    maxindex = np.where(errors == maxerror)
    optuning = alphas[maxindex]


    lasso.set_params(alpha=optuning[0])
    lasso.fit(X_train, y_train)
    lasso_coef = lasso.coef_
    lasso_coef = std_Y*lasso_coef/std_X;
    lasso_mse = np.dot(np.dot(lasso_coef.T-beta.T, Temp),lasso_coef-beta)
    
    Lasso_MSE_log.append(lasso_mse[0,0])
    Lasso_df_log.append(len(set(lasso_coef[np.nonzero(lasso_coef)])))
    
    #######################################################
    
    
    alphas1 = np.arange(0.01,1,0.05)
    alphas2 = np.arange(0.01,1.5,0.05)
    elnet = ElasticNet(random_state=0)
    coefs = []
    errors =[]
    optuning=np.array([0,0])
    besterror = -100

    for a in alphas1:
        for b in alphas2:
            elnet.set_params(alpha=a+b, l1_ratio=a/(a+b))
            elnet.fit(X_train, y_train)
            coefs.append(elnet.coef_)
            temperror = elnet.score(X_test, y_test)
            if temperror > besterror:
                besterror = temperror
                optuning=np.array([a,b])
        


    elnet.set_params(alpha=optuning[0]+optuning[1], l1_ratio=optuning[0]/(optuning[0]+optuning[1]))
    elnet.fit(X_train, y_train)
    en_coef = elnet.coef_
    en_coef = std_Y*en_coef/std_X;
    en_mse = np.dot(np.dot(en_coef.T-beta.T, Temp),en_coef-beta)
    
    Elastic_net_MSE_log.append(en_mse[0,0])
    Elastic_net_df_log.append(len(set(en_coef[np.nonzero(en_coef)])))
    
    
    print(percentage, '%');

np.savetxt('OSCAR_MSE_log2.txt', OSCAR_MSE_log)

np.savetxt('Lasso_MSE_log2.txt', Lasso_MSE_log)

np.savetxt('Elastic_net_MSE_log2.txt', Elastic_net_MSE_log)

np.savetxt('OSCAR_df_log2.txt', OSCAR_df_log)

np.savetxt('Lasso_df_log2.txt', Lasso_df_log)

np.savetxt('Elastic_net_df_log2.txt', Elastic_net_df_log)

np.savetxt('clf_MSE_log2.txt', clf_MSE_log)

np.savetxt('clf_df_log2.txt', clf_df_log)


print('OSCAR MSE:', np.mean(OSCAR_MSE_log))
print('Ridge MSE:', np.mean(clf_MSE_log))
print('Lasso MSE:', np.mean(Lasso_MSE_log))
print('Elastic Net MSE:', np.mean(Elastic_net_MSE_log))
















