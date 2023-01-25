#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Black Carbon concentration proxy using simple ML algorithms
Created on Fri Jan 20 12:33:20 2023

@author: jparedes
"""
import time
import os
import argparse

import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import pickle

from sklearn.preprocessing import StandardScaler


from sklearn.metrics import mean_squared_error, r2_score

from sklearn.pipeline import Pipeline
from sklearn import svm, tree
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor


import matplotlib as mpl
import matplotlib.pyplot as plt

# modules
import Load_dataSet as LDS
#%%

parser = argparse.ArgumentParser(
    prog = 'Black Carbon Temporal proxy using ML models',
    description='Train Black Carbon concentration proxy using a set of ML models',
    epilog = '--'
    )
# data set related
parser.add_argument('-d','--device',help='Device used for obtaining pollutant measurements.',choices=['Ref_st','LCS'],default='Ref_st')
parser.add_argument('-s','--datasource',help='Source of data set. Can be either raw .csv files or processed pandas data frame',choices=['raw','data_frame'],default='data_frame')
# proxy methodology
parser.add_argument('-p','--predictors',nargs='+',action='append',help='<Required> predictors for BC proxy. Available pollutants: BC,N, PM1, PM25,PM10,O3,NO2,NO,SO2,T,RH,Vmax',required=True)
parser.add_argument('-m','--monthsTrain',type=int,help='Number of months used for training the neural network',default=12)
parser.add_argument('-v','--monthsVal',type=int,help='Number of months used for validating the model during training optimization',default=6)
parser.add_argument('-sc','--scaler',help='Feature scaling procedure',choices=['Standard','MinMax'],default='Standard')
parser.add_argument('-fh','--forecastHours',type=int,help='Number of hours to forecast. 0 means no forecasting and the model predicts the last measurement of the sequence')
parser.add_argument('--train', help='Use it to specify training a ML algorithm',action='store_true')

#filenames
parser.add_argument('-gsfn','--gridSearchFileName',type=str,help='File name of data frame containing sorted grid search results',default='GridSearchResults_BCproxy')
parser.add_argument('-pfn','--predictionsFileName',type=str,help='File name of BC proxy model predictions picke file')
parser.add_argument('-smfn','--scoringMetricsFileName',type=str,help='File name of BC proxy model scoring metrics report')
parser.add_argument('-mla','--algorithm',type=str,help='ML algorithm used to fit data',default='SVR',choices=['MLR','SVR','RF','MLP','AdaBoost_SVR','AdaBoost_DT'],required=True)

# arguments for testing a specific ML algorithm
## SVR
parser.add_argument('-C',type=float,help='SVR C hyperparameter value')
parser.add_argument('-e','--epsilon',type=float,help='SVR epsilon hyperparameter value')
parser.add_argument('-g','--gamma',type=float,help='SVR gamma RBF kernel hyperparameter')

## RF/DT
parser.add_argument('-ne','--nEstimators',type=int,help='RF (number of trees) and AdaBoost (number of iterations)  hyperparameter')
parser.add_argument('-md','--maxDepth',type=int,help='RF and DT algorithm max depth hyperparameter')
parser.add_argument('-mss','--minSamplesSplit',type=int,help='RF algorithm minimum number of samples for split hyperparameter')
parser.add_argument('-ms','--maxSamples',type=float,help='RF algorithm fraction of maximum number of samples used hyperparameter')
parser.add_argument('-mf','--maxFeatures',type=float,help='RF algorithm maximum number of features to use hyperparameter')

## MLP
parser.add_argument('-hs','--hiddenSize',type=int,help='MLP number of hidden elements per hidden layer',nargs='+')        
parser.add_argument('-reg','--regularization',type=float,help='MLP algorithm L2 regularization hyperparameter')
parser.add_argument('-lr','--learningRate',type=float,help='MLP and AdaBoost learning rate hyperparameter')

## AdaBoost
parser.add_argument('-l','--loss',type=str,help='AdaBoost loss function',choices=['square','exponential'])
parser.add_argument('-k','--kernel',type=str,help='Adaboost SVR base estimator kernel hyperparameter',choices=['poly','rbf'])
parser.add_argument('-msl','--minSamplesLeaf',type=int,help='DT hyperparameter. Number of samples to be left in a leaf node')

# hardware related
parser.add_argument('-nj','--numberJobs',type=int,help='<Sklearn required> number of processors used for fitting model', default=2)
parser.add_argument('-pd','--preDispatch',type=int,help='<Sklearn required> number of pre-dispatched jobs', default=2)


args = parser.parse_args()

#%% data set
def prepareDataSet(args,file_path):
    """
    Prepare time sequences for model

    Parameters
    ----------
    args : parser
        arguments from input. The sequence length and sensor type are obtained from here
    file_dir : str
        path to save generated data frame
    file_name : str
        name of genrated data frame
    save_file : bool
        save generated data set. Can be disabled for short testing purposes

    Returns
    -------
    ds : obj
        data set containing tabular data and sequences for train/validation/test sets

    """
    # create data set
    ds = LDS.dataSet(files_path=file_path,device=args.device,source=args.datasource)
    ds.load_dataSet()
    # select variables for proxy
    df = ds.df.loc[:,args.predictors[0]]
    # tokenize
    ## train/val/test split
    ds = LDS.dataSet_split(df,train=args.monthsTrain,val=args.monthsVal)
    ds.train_val_test_split()
    ## scale data set
    ds.Scale(scaler=args.scaler)
    return ds

def model_fit(pipe,grid_params,X_train,Y_train,X_val,Y_val,args):
    """
    Grid search on training set and validating on fixed validation set
    """
    if args.algorithm == 'SVR':
        results = []
        results.append(['RMSE_train','R2_train','RMSE_val','R2_val','C','gamma','epsilon'])
        for C in grid_params['model__C']:
            for g in grid_params['model__gamma']:
                for e in grid_params['model__epsilon']:
                    pipe.named_steps['model'].C = C
                    pipe.named_steps['model'].gamma = g
                    pipe.named_steps['model'].epsilon = e
                    # fit on training
                    pipe.fit(X_train,Y_train)
                    # traning scoring
                    y_pred_train,RMSE_train,R2_train = BC_proxy_prediction(pipe,X_train,Y_train)
                    # validation scoring
                    y_pred_val,RMSE_val,R2_val = BC_proxy_prediction(pipe,X_val,Y_val)
                    
                    # save results
                    results.append([RMSE_train,R2_train,RMSE_val,R2_val,C,g,e])
    
    elif args.algorithm == 'RF':
        results = []
        results.append(['RMSE_train','R2_train','RMSE_val','R2_val','n_estimators','max_features','max_depth','min_samples_split','max_samples'])
        
        for n_estimators in grid_params['model__n_estimators']:
            for max_features in grid_params['model__max_features']:
                for max_depth in grid_params['model__max_depth']:
                    for min_samples_split in grid_params['model__min_samples_split']:
                        for max_samples in grid_params['model__max_samples']:
                            pipe.named_steps['model'].n_estimators = n_estimators
                            pipe.named_steps['model'].max_features = max_features
                            pipe.named_steps['model'].max_depth = max_depth
                            pipe.named_steps['model'].min_samples_split = min_samples_split
                            pipe.named_steps['model'].max_samples = max_samples

                            # fit on training
                            pipe.fit(X_train,Y_train)
                            # traning scoring
                            y_pred_train,RMSE_train,R2_train = BC_proxy_prediction(pipe,X_train,Y_train)
                            # validation scoring
                            y_pred_val,RMSE_val,R2_val = BC_proxy_prediction(pipe,X_val,Y_val)
                            
                            # save results
                            results.append([RMSE_train,R2_train,RMSE_val,R2_val,n_estimators,max_features,max_depth,min_samples_split,max_samples])
                            
    elif args.algorithm == 'MLP':
        results = []
        results.append(['RMSE_train','R2_train','RMSE_val','R2_val','lr_init','hidden_layer_sizes','alpha'])    
            
        for lr in grid_params['model__learning_rate_init']:
            for hls in grid_params['model__hidden_layer_sizes']:
                for alpha in grid_params['model__alpha']:
                    pipe.named_steps['model'].learning_rate_init = lr
                    pipe.named_steps['model'].hidden_layer_sizes = hls
                    pipe.named_steps['model'].alpha = alpha

                    # fit on training
                    pipe.fit(X_train,Y_train)
                    # traning scoring
                    y_pred_train,RMSE_train,R2_train = BC_proxy_prediction(pipe,X_train,Y_train)
                    # validation scoring
                    y_pred_val,RMSE_val,R2_val = BC_proxy_prediction(pipe,X_val,Y_val)
                            
                    # save results
                    results.append([RMSE_train,R2_train,RMSE_val,R2_val,lr,hls,alpha])


                        
        return results
    
#%%
# GridSearch
def BC_proxy_fit(X_train,Y_train,X_val,Y_val,args):
    # grid search fitting models on training set using a fixed validation set
    n_features = X_train.shape[1]
    
    if args.algorithm=='SVR':
        print(f'---------------\nFitting {args.algorithm}\n----------------')
        grid_params = {
            'model__C':np.logspace(-3,3,7),
            'model__gamma':np.logspace(-3,3,7),
            'model__epsilon':np.linspace(0.1,0.8,5)
            }
        model = svm.SVR(kernel='rbf')
        pipe = Pipeline([('model', model)])
            
    elif args.algorithm=='RF':
        print(f'---------------\nFitting {args.algorithm}\n----------------')
        model = RandomForestRegressor(criterion='squared_error', min_samples_leaf=2,max_leaf_nodes=None,
                                      bootstrap=True, oob_score=True, random_state=92,
                                      n_jobs=1, verbose=1, warm_start=False)

        grid_params = {
            'model__n_estimators':[100,500,1000,3000],
            'model__max_features':[1.0,0.5],
            'model__max_depth':[20,10,5],
            'model__min_samples_split':[10,2],
            'model__max_samples':[1.0,0.5,0.33]
            }   
    
        pipe = Pipeline(steps=[('model', model)])


    elif args.algorithm=='MLP':
        print(f'---------------\nFitting {args.algorithm}\n----------------')
        
        model = MLPRegressor(solver='adam',learning_rate='adaptive', activation='tanh',batch_size=128,
                             max_iter=5000,shuffle=False,random_state=92,warm_start=False,
                             tol=1e-4,early_stopping=True,validation_fraction=0.1,n_iter_no_change=10)
        grid_params = {
            'model__learning_rate_init':np.logspace(-4,-2,3),
            'model__hidden_layer_sizes':[(2*n_features,),
                                         (n_features,),
                                         (int(0.5*n_features),),
                                         (2*n_features,2*n_features),
                                         (n_features,n_features),
                                         (int(0.5*n_features),int(0.5*n_features)),
                                         (n_features,n_features,n_features),
                                         (50,)],
            'model__alpha':np.logspace(-3,-1,3),
            }
        
        pipe = Pipeline([('model', model)])
        
    elif args.algorithm == 'AdaBoost_SVR':
        print(f'---------------\nFitting {args.algorithm}\n----------------')
        
        # base models
        
        scaler = StandardScaler()
        base_estimator = svm.SVR(epsilon=0.1,degree=3)
        grid_params = {
            'model__base_estimator__kernel':['poly','rbf'],
            'model__base_estimator__C': [0.01,0.1,1.0],
            'model__base_estimator__gamma':[0.1,1.0],
            'model__n_estimators':[10,50],
            'model__learning_rate':[0.01,0.1,0.5],
            'model__loss':['square','exponential']
            }
        model = AdaBoostRegressor(base_estimator=base_estimator,random_state=92)
        pipe = Pipeline([('scaler',scaler),('model', model)])
            
    elif args.algorithm == 'AdaBoost_DT':
        print(f'---------------\nFitting {args.algorithm}\n----------------')
        base_estimator = tree.DecisionTreeRegressor(criterion='squared_error',splitter='best',
                                                    min_samples_split=2,max_features=None)
        grid_params={
            'model__base_estimator__max_depth':[5,50,100],
            'model__base_estimator__min_samples_leaf':[0.5,1],
            'model__base_estimator__min_samples_split':[0.5,2],
            'model__n_estimators':[10,50,200],
            'model__learning_rate':[0.01,0.1,0.5],
            'model__loss':['square','exponential']
            }
            
        model = AdaBoostRegressor(base_estimator=base_estimator,random_state=92)
        pipe = Pipeline([('model', model)])
            
        
        
        

    # gridsearch
    start_time = time.time()
    validation_results = model_fit(pipe,grid_params,X_train,Y_train,X_val,Y_val,args)
    end_time = time.time()
    
    print('Grid search finished in %.2f'%(end_time-start_time))
    
    return validation_results

#%% Create a specific ML model
def BC_proxy_model(args):
    # Loads a ML proxy model.
    # Can specify the hyperparameters found via gridsearch 
    if args.algorithm =='SVR':
        
        C = args.C
        epsilon = args.epsilon
        gamma = args.gamma
        
        model = svm.SVR(kernel='rbf',C=C, epsilon=epsilon, gamma=gamma)
        model_BC = Pipeline([('model', model)])
            
    elif args.algorithm == 'RF':
        
        n_estimators = args.nEstimators
        max_depth = args.maxDepth
        min_samples_split = args.minSamplesSplit
        max_samples = args.maxSamples
        max_features = args.maxFeatures
        
        model = RandomForestRegressor(criterion='squared_error', min_samples_leaf=2,max_leaf_nodes=None,
                                      bootstrap=True, oob_score=False, random_state=92,
                                      n_jobs=1, verbose=1, warm_start=False,
                                      n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split,
                                      max_samples=max_samples,max_features=max_features)
        model_BC = model
    
    elif args.algorithm == 'MLP':
        
        lr = args.learningRate
        hidden_layer_sizes = tuple(args.hiddenSize)
        alpha = args.regularization
        
        
        model = MLPRegressor(solver='adam',learning_rate='adaptive',
                             max_iter=5000,shuffle=True,random_state=92,warm_start=False,
                             batch_size=128,early_stopping=True,
                             learning_rate_init=lr,tol=1e-6,activation='tanh',
                             alpha=alpha,hidden_layer_sizes=hidden_layer_sizes
                             )
        model_BC = Pipeline([('model', model)])
        
    elif args.algorithm == 'AdaBoost_SVR':
        
        
        kernel = args.kernel
        C = args.C
        gamma = args.gamma
        nEstimators = args.nEstimators
        lr = args.learningRate
        loss = args.loss
            
        scaler = StandardScaler()
        base_estimator = svm.SVR(epsilon=0.1,degree=3,
                                 kernel=kernel,C=C,gamma=gamma)
        model = AdaBoostRegressor(base_estimator=base_estimator,random_state=92,
                                  n_estimators=nEstimators,learning_rate=lr,loss=loss)
        model_BC = Pipeline([('scaler',scaler),('model', model)])
            
    elif args.algorithm == 'AdaBoost_DT':
        maxDepth = args.maxDepth
        minSamplesLeaf = args.minSamplesLeaf
        nEstimators = args.nEstimators
        lr = args.learningRate
        loss = args.loss
            
            
        base_estimator = tree.DecisionTreeRegressor(criterion='squared_error',splitter='best',
                                                    min_samples_split=2,max_features=None,
                                                    max_depth=maxDepth,min_samples_leaf=minSamplesLeaf)
            
        model = AdaBoostRegressor(base_estimator=base_estimator,random_state=92,
                                  n_estimators=nEstimators,learning_rate=lr,loss=loss)
        model_BC = Pipeline([('model', model)])

            
            
        
    
    print(f'Loading BC proxy model\n{model_BC}')

    return model_BC
#%%
def BC_proxy_prediction(BC_model,X,Y,scaler):
    
    # model prediction and scoring
    y_pred = BC_model.predict(X)
    
    if args.scaler =='Standard':
        descaler = StandardScaler()
        descaler.mean_ = scaler.mean_[0]
        descaler.scale_ = scaler.scale_[0]

    RMSE = np.sqrt(mean_squared_error(descaler.inverse_transform(Y.values.reshape(-1,1)),
                                      descaler.inverse_transform(y_pred.reshape(-1,1))))
    R2 = r2_score(descaler.inverse_transform(Y.values.reshape(-1,1)),
                                     descaler.inverse_transform(y_pred.reshape(-1,1)))
    #adj_R2 = 1-(1-R2)*(X.shape[0]-1)/(X.shape[0]-X.shape[1]-1)
    
    #print(f'BC proxy model predictions\n RMSE = {RMSE}\t R2 = {R2}\t adj-R2 = {adj_R2}\n')
    
    return y_pred,RMSE,R2

def saveResults(model,path,y_pred_train,Y_train,y_pred_test,Y_test,RMSE_train,R2_train,RMSE_test,R2_test,args):

    # save model scoring metrics
    if not args.scoringMetricsFileName:
        fname = 'ScoringMetrics_BCproxy_'+args.algorithm+'.txt'
    else:
        fname = args.scoringMetricsFileName
    dt = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    f = open(path+'/'+fname,'a')
    print(15*'-',file=f)
    print(dt,file=f)
    print(f'BC model:\n {model}',file=f)
    print(f'Training scoring\nRMSE = {RMSE_train}\nR2 = {R2_train}',file=f)
    print(f'Testing scoring\nRMSE = {RMSE_test}\nR2 = {R2_test}',file=f)
    f.close()

    # save model predictions
    if not args.predictionsFileName:
        fname = 'Predictions_BCproxy_'+args.algorithm+'_'+dt+'.pkl'
    else:
        fname = args.predictionsFileName
        
    df_train = pd.concat([pd.DataFrame(y_pred_train,index=Y_train.index),Y_train,pd.DataFrame(y_pred_test,index=Y_test.index),Y_test],axis=1)
    df_train.columns = ['Train_pred','Train_true','Test_pred','Test_true']
    df_train.to_pickle(path+'/'+fname)
    print(f'{df_train.head()}')

    # save model itself
    fname = 'Model_BCproxy_'+args.algorithm+'_'+dt+'.pkl'
    
    with open(path+'/'+fname,'wb') as f:
        pickle.dump(model,f)
        
    # save predictions scatter plot
    fname = path+'/'+'BCproxy_'+args.algorithm+'_'+dt+'_scatterPlot.pdf'
    plotPrediction(df_train,RMSE_test,fname,args)
    
    print(f'Scoring metrics, predictions and model saved on\n {path}')
    
    
#%% Compare prediction and real value
def plotPrediction(df,RMSE_test,fname,args):
    # Pollutant time series
    fig,ax = plt.subplots(figsize=(20,10),nrows=1,ncols=1)
    l = 'RMSE = '+str(np.round(RMSE_test,2))+' ($\mu$g/m$^3$)'
    ax.scatter(df.loc[:,'Test_true'],df.loc[:,'Test_pred'],color='#1F618D',label=l)
    ax.plot(df.loc[:,'Test_true'],df.loc[:,'Test_true'],color='#A93226',linestyle='--')

    ymin = 0.0
    ymax = df.loc[:,'Test_pred'].max()+1
    yrange=np.ceil(np.arange(ymin,ymax,10))
    ylabels = [str(int(i)) for i in yrange]
    ax.set_yticks(ticks=yrange)
    plt.ylim(ymin,ymax)
    
    ylabel = 'Predicted BC concentration ($\mu$g/m$^3$)'
    ax.set_yticklabels(labels=ylabels,fontsize=30)
    ax.set_ylabel(ylabel,rotation=90,fontsize=30)
    
    xmin = 0.0
    xmax = df.loc[:,'Test_true'].max()+1
    xrange=np.ceil(np.arange(xmin,xmax,10))
    xlabels = [str(int(i)) for i in xrange]
    ax.set_xticks(ticks=xrange)
    plt.xlim(xmin,xmax)
    
    xlabel = 'Real BC concentration ($\mu$g/m$^3$)'
    ax.set_xticklabels(labels=xlabels,fontsize=30) 
    ax.set_xlabel(xlabel,rotation=0,fontsize=30)
    
    ax.tick_params(direction='out', length=4, width=1)
    ax.tick_params(axis='both', which='major')
    ax.legend(loc='upper right',prop={'size':30,'weight':'normal'},ncol=1,framealpha=0.3,edgecolor = 'black',
               handleheight = 1,handletextpad=0.2)
    fig.suptitle(f'BC real vs {args.algorithm} predicted concentrations\nTesting set', fontsize=30)
    ax.grid(alpha=0.5)
    
    
    plt.savefig(fname,dpi=600,format='pdf')


#%% main
def main(ds,args,training_path,results_path):
    """
    Fit a ML algorithm
    The object data set already has a training and validation set.
    The first column is the target and the others are the predictors
    NO CROSS VALIDATION BUT A VALIDATION SET
    """
    
    df_train = ds.train_set_scaled
    df_val = ds.val_set_scaled
    X_train = df_train.iloc[:,1:]
    Y_train = df_train.iloc[:,0]
    X_val = df_val.iloc[:,1:]
    Y_val = df_val.iloc[:,0]
    
    if args.train:# perform hyperparameters optimization
        input(f'Starting Grid Search CV hyperparameters optimization for {args.algorithm}\nPress Enter to continue...')
        validation_results = BC_proxy_fit(X_train,Y_train,X_val,Y_val,args)
        results = pd.DataFrame(validation_results)
        # save results
        fname = args.gridSearchFileName+'_'+args.algorithm+'.pkl'
        results.to_pickle(training_path+'/'+fname)
        
        fname = args.gridSearchFileName+'_'+args.algorithm+'.csv'
        results.to_csv(training_path+'/'+fname)
        print(f'Grid search results saved on {training_path} as {fname}')

    else:# Not training a model but predicting using a pre-trained one
        
        df_test = ds.test_set_scaled
        X_test = df_test.iloc[:,1:]
        Y_test = df_test.iloc[:,0]
        print(f'Testing ML algorithm {args.algorithm} on Testing set of {Y_test.shape[0]} samples from {Y_test.index[0]} until {Y_test.index[-1]}')
        
        # create BC model given hyperparameters
        BC_model = BC_proxy_model(args)
        BC_model.fit(X_train,Y_train)
        print('Training set scoring')
        y_pred_train, RMSE_train, R2_train  = BC_proxy_prediction(BC_model,X_train,Y_train,ds.scaler)
        print('Testing set scoring')
        y_pred_test, RMSE_test, R2_test  = BC_proxy_prediction(BC_model,X_test,Y_test,ds.scaler)
        # save results
        saveResults(BC_model,results_path,y_pred_train,Y_train,y_pred_test,Y_test,RMSE_train,R2_train,RMSE_test,R2_test,args)

    
    return

if __name__=='__main__':
    abs_path = os.path.dirname(os.path.realpath(__file__))
    # paths for training
    file_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Files'
    training_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Results/'+args.algorithm
    results_path= os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Results/'+args.algorithm+'/predictions'
    os.chdir(abs_path)
    
    # load ds
    ds = prepareDataSet(args,file_path=file_path)
    # fit or predict: depending on args.train
    main(ds,args,training_path,results_path)
    
    
    
    


