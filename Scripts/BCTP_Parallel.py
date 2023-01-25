#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BC concentration Proxy Forecasting
The original machine has no gpu

Created on Fri Nov  4 11:47:02 2022

@author: jparedes
"""

import time
import os
import argparse

import numpy as np
import math
import pandas as pd
from tqdm import tqdm
import pickle

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pack_padded_sequence,pad_packed_sequence
from torch import nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping,TQDMProgressBar,RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import AdvancedProfiler
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.demos.boring_classes import DemoModel, BoringDataModule

from sklearn.metrics import mean_squared_error, r2_score
#from pytorch_lightning.utilities.model_summary import ModelSummary

# modules
import Load_dataSet as LDS
import Plots

#%%
def script_params():
    parser = argparse.ArgumentParser(
        prog = 'Black Carbon Forecasting Proxy',
        description='Train Time Forecasting BC concentration using Neural Networks',
        epilog = '--'
        )
    # data set related
    parser.add_argument('-d','--device',help='Device used for obtaining pollutant measurements.',choices=['Ref_st','LCS'],default='Ref_st')
    parser.add_argument('-s','--datasource',help='Source of data set. Can be either raw .csv files or processed pandas data frame',choices=['raw','data_frame'],default='data_frame')
    parser.add_argument('-pd','--proxydataset',help='Filename of pickle file to save the predictors temporal sequence used for current BC proxy experiment',default='experiment_dataSet.pkl')
    parser.add_argument('-f','--forecast',help='Specify if the prediction is forecasting or just the last measurement of the sequence length',action='store_true')
    
    # proxy methodology
    parser.add_argument('-p','--predictors',nargs='+',action='append',help='<Required> predictors for BC proxy. Available pollutants: BC,N, PM1, PM25,PM10,O3,NO2,NO,SO2,T,RH,Vmax',required=True)
    parser.add_argument('-m','--monthsTrain',type=int,help='Number of months used for training the neural network',default=12)
    parser.add_argument('-v','--monthsVal',type=int,help='Number of months used for validating the model during training optimization',default=6)
    parser.add_argument('-sl','--seqlength',type=int,help='Number of previous measurements considered in the time window',default=2)
    parser.add_argument('-sc','--scaler',help='Feature scaling procedure',choices=['Standard','MinMax'],default='Standard')
    parser.add_argument('-fh','--forecastHours',type=int,help='Number of hours to forecast. 0 means no forecasting and the model predicts the last measurement of the sequence')
    
    # RNN model
    parser.add_argument('--train', help='Use it to specify training a Neural Netowrk',action='store_true')
    parser.add_argument('-a','--architecture',help='Architecture of Artificial Neural Network',choices=['RNN','LSTM','GRU','TDNN'],required=True)
    parser.add_argument('-k','--kernelsize',type=int,help='Size of Kernel for TDNN (1D-CNN)',default=2)
    parser.add_argument('-hl','--hiddenlayers',type=int,help='Number of stacked hidden layers',default=1)
    parser.add_argument('-nhl','--nodeshl',type=int,help='Number of nodes in hidden layers. The number of neurons is the same for all layers',default=2)
    parser.add_argument('-o','--noutput',type=int,help='Number of output neurons',default=1)
    parser.add_argument('-e','--epochs',type=int,help='Maximum number of epochs for training. The process stops after reaching this number',default=5000)
    parser.add_argument('-lr','--learningrate',type=float,help='Learning rate for optimizer',default=1e-4)
    parser.add_argument('-wd','--weightDecay',type=float,help='L2 normalization for RNNs',default=1e-2)
    parser.add_argument('-bs','--batchsize',type=int,help='Number of measurements per batch',default=128)
    parser.add_argument('-dop','--dropoutprob',type=float,help='Probability of dropout',default=0.0)
    parser.add_argument('-ar','--autoRegressive',help='Bool for autoregressive model (include target as input)',action='store_true',default=False)
    
    
    # loading trained model
    parser.add_argument('-ese','--earlyStoppingEpoch',type=int,help='Epoch at which a trined model stopped. The number is reported for trained model in checkpoints')
    parser.add_argument('-tl','--trainLoss',type=str,help='Training loss of trained model. The value is reported for trained model in checkpoints')
    parser.add_argument('-vl','--validationLoss',type=str,help='Validation loss of trained model. The value is reported for trained model in checkpoints')
    
    # hardware related
    parser.add_argument('-acc','--accelerator',help='Accelerator for training',default='cpu',choices=['cpu','gpu'])
    parser.add_argument('-nd','--ndevices',type=int,help='Number of devices for training',default=1)
    parser.add_argument('-nw','--numworkers',type=int,help='Number of devices for training',default=0)
    args = parser.parse_args()
    
    if args.autoRegressive:
        args.npredictors = len(args.predictors[0])
    else:
        args.npredictors = len(args.predictors[0])-1
        
    return args
#%% Pytorch DataSet
def prepareDataSet(args,file_path,file_name='proxyDataSet.pkl',save_file=True):
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
    # create forecasting time window
    ds.create_sequences_dates(input_set = 'Train', target_column='BC', sequence_length=args.seqlength,auto_regressive=args.autoRegressive,n_hours_forecast=args.forecastHours)
    ds.create_sequences_dates(input_set = 'Validation', target_column='BC', sequence_length=args.seqlength,auto_regressive=args.autoRegressive,n_hours_forecast=args.forecastHours)
    ds.create_sequences_dates(input_set = 'Test', target_column='BC', sequence_length=args.seqlength,auto_regressive=args.autoRegressive,n_hours_forecast=args.forecastHours)
        
    #save to disk
    if save_file:
        with open(file_path+'/Proxy_dataSet/'+file_name,'wb') as handle:
            pickle.dump(ds,handle)
    return ds

class BC_DataSet(Dataset):
    def __init__(self,sequences):
        self.sequences = sequences
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self,idx):
        sequence,label = self.sequences[idx]
        return dict(
            sequence = torch.Tensor(sequence.to_numpy()),
            label = torch.from_numpy(np.asarray(label))
            )

class BC_DataModule(pl.LightningDataModule):
    def __init__(self,ds,args):
        """
        BC dataModule
        
        Parameters
        ----------
        proxy_predictors : array
            Array containing the name of parameters
        device_load : str
            Type of sensor used for obtaining the measurements: RefSt or LCS
        data_source : str
            How are measurements stored. Raw data or pandas dataFrames
        n_months_traininig : int
            number of months used for training ANN
        n_months_val : int
            number of months left for validating the ANN during training
        batch_size : int
            number of elements per batch
        sequence_length : int
            time window length of past measurements
        fname : str
            name of the file containing the data set for this run
        file_dir : str
            directory for saving the data files for this run

        Returns
        -------
        None.

        """
        super().__init__()
        
        self.train_sequences = ds.train_set_sequence_dates
        self.val_sequences = ds.val_set_sequence_dates
        self.test_sequences = ds.test_set_sequence_dates
        self.batch_size = args.batchsize
        self.numworkers = args.numworkers

        
        
    
    # def prepare_data(self):
    #     # create data set
    #     ds = LDS.dataSet(device=self.sensor_device,source=self.data_source)
    #     ds.load_dataSet()
    #     # select variables for proxy
    #     df = ds.df.loc[:,self.predictors]
        
    #     # tokenize
    #     ## train/val/test split
    #     ds = LDS.dataSet_split(df,train=self.n_months_training,val=self.n_months_val)
    #     ds.train_val_test_split()
    #     ## scale data set
    #     ds.Scale(scaler=self.scaler)
    #     # create forecasting time window    
    #     ds.create_sequences(input_set = 'Train', target_column='BC', sequence_length=self.sequence_length)
    #     ds.create_sequences(input_set = 'Validation', target_column='BC', sequence_length=self.sequence_length)
    #     ds.create_sequences(input_set = 'Test', target_column='BC', sequence_length=self.sequence_length)
        
    #     #save to disk
    #     with open(self.file_dir+self.file_name,'wb') as handle:
    #         pickle.dump(ds,handle)
        

    def setup(self,stage:str):
        # # load back data
        # with open(self.file_dir+self.file_name,'rb') as handle:
        #     ds = pickle.load(handle)
        
        # assign splits for DataLoaders
        if stage == 'fit':
            self.train_dataset = BC_DataSet(self.train_sequences)
            self.val_dataset = BC_DataSet(self.val_sequences)
        if stage == 'predict':
            self.test_dataset = BC_DataSet(self.test_sequences)
    
    def train_dataloader(self):
        # def collate_(batch):
        #     x = [item['sequence'] for item in batch]
        #     y = [item['label'] for item in batch]
        #     # x_len = [len(x) for x in x_item]
        #     # y_len = [len(y) for y in y_item]
        #     # x_pad = pad_sequence(x_item,batch_first=True,padding_value=0)
        #     # y_pad = pad_sequence(y_item,batch_first=True,padding_value=0)
        #     return x,y
        # #sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset,shuffle=False)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.numworkers
            #collate_fn=collate_
            # sampler=sampler
            )
    def val_dataloader(self):
        #sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset,shuffle=False)
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle = False,
            num_workers = self.numworkers
            # sampler=sampler
            )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.numworkers
            )

#%% Temporal BC proxy: TBCP

class BC_Forecasting_Proxy(nn.Module):
    
    def __init__(self,args):
        super().__init__()
        self.n_hidden = args.nodeshl
        self.architecture = args.architecture
        
        # time step pass
        if args.architecture == 'RNN':
            self.network = nn.RNN(
                input_size=args.npredictors,
                hidden_size = args.nodeshl,
                batch_first = True,
                num_layers = args.hiddenlayers,
                nonlinearity = 'relu',
                bias = True,
                dropout = args.dropoutprob,
                bidirectional = False
                )
            self.regressor = nn.Linear(args.nodeshl,args.noutput,bias=True)
            
        elif args.architecture == 'LSTM':
            self.network = nn.LSTM(
                input_size= args.npredictors,
                hidden_size = args.nodeshl,
                batch_first = True,
                num_layers = args.hiddenlayers,
                bias = True,
                dropout = args.dropoutprob,
                bidirectional=False
                )
            self.regressor = nn.Linear(args.nodeshl,args.noutput,bias=True)
            
        elif args.architecture == 'GRU':
            self.network = nn.GRU(
                input_size = args.npredictors,
                hidden_size = args.nodeshl,
                batch_first = True,
                num_layers = args.hiddenlayers,
                bias = True,
                dropout = args.dropoutprob,
                bidirectional = False
                )
            self.regressor = nn.Linear(args.nodeshl,args.noutput,bias=True)
            
        elif args.architecture == 'TDNN':
            self.network = nn.Conv1d(
                in_channels = args.npredictors,
                out_channels = args.noutput,
                kernel_size = args.kernelsize,
                stride = 1,
                padding = 0,
                dilation= 1,
                groups = 1,
                bias = True)
            self.activation_function = nn.ReLU()
            self.regressor = nn.Linear(args.seqlength-args.kernelsize+1,args.noutput,bias=True)
            
        
        
    def forward(self,x):
        # Input.size = (batch_size, sequence_length, n_predictors)
        #self.rnn.flatten_parameters()# distributed training
        
        # time step prediction
        if self.architecture == 'TDNN':
            x = x.transpose(1,2).contiguous()
            output = self.network(x)
            output = self.activation_function(output)
            return self.regressor(output)
            
        elif self.architecture in ['RNN','GRU']:
            output,h_n = self.network(x)
            return self.regressor(output[:,-1,:])
        else:
            output,(h_n,c_n) = self.network(x)
            return self.regressor(output[:,-1,:])
        
        
class BC_Predictor(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        self.model  = BC_Forecasting_Proxy(args)
        self.criterion = nn.MSELoss()
        self.lr = args.learningrate
        self.weightDecay = args.weightDecay
        self.save_hyperparameters()
        
    
    def forward(self,x,labels=None):
        output = self.model(x)
        output.to(x)
        loss=0
        if labels is not None:
            loss = self.criterion(output.squeeze().float(),labels.squeeze().float())
        return loss,output
    
    def training_step(self,batch,batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        # sequences = pack_sequence(batch[0],enforce_sorted=False)
        # labels = pack_sequence(batch[1],enforce_sorted=False)
        loss,outputs = self(sequences,labels) 
        #self.log('train_loss',loss,prog_bar=True,logger=True,on_step=False,on_epoch=True,enable_graph=True)
        self.log_dict({'train_loss':loss,'step':self.current_epoch+1},prog_bar=True,logger=True,on_step=False,on_epoch=True,enable_graph=True)
        self.logger.log_graph(self)
        return {'loss':loss}
    
    def validation_step(self,batch,batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        loss,outputs = self(sequences,labels)
        #self.log('val_loss',loss,prog_bar=True,logger=True,on_step=False,on_epoch=True)
        self.log_dict({'val_loss':loss,'step':self.current_epoch+1},prog_bar=True,logger=True,on_step=False,on_epoch=True,enable_graph=True, sync_dist=True)
        self.logger.log_graph(self)
        return loss
    
    def test_step(self,batch,batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        loss,outputs = self(sequences,labels)
        #self.log('test_loss',loss,prog_bar=True,logger=True,on_step=False,on_epoch=True)
        self.log_dict({'test_loss':loss,'step':self.current_epoch+1},prog_bar=True,logger=True,on_step=False,on_epoch=True,enable_graph=True, sync_dist=True)
        self.logger.log_graph(self)
        return loss
    
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(),lr=self.lr, weight_decay = self.weightDecay)
    
class ProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.set_description('running traininig ...')
        return bar
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description('running validation ...')
        return bar
    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        bar.set_description('running testing ...')
        return bar

#%% Testing phase functions
def modelPredictions(args,ds,data_module,trained_model):
    """
    BC predictions from model
    Complementary to Trainer() results
    """
    print('Predicting concentration based on model')
    # reverse transformation: get scaling parameters obtained from training set
    if args.scaler == 'Standard':
        descaler = StandardScaler()
        descaler.mean_ = ds.scaler.mean_[0]
        descaler.scale_ = ds.scaler.scale_[0]
    
    else:
        descaler = MinMaxScaler()
        descaler.min_ = ds.scaler.min_[0]
        descaler.scale_ = ds.scaler.scale_[0]
   
  
    # predictions
    predictions = []
    true_val = []
    
    

    for item in tqdm(data_module.test_dataloader()):#data_module.test_dataloader() or data_module.test_dataset
        sequence = item['sequence']
        label = item['label']
        _, pred = trained_model(sequence)
        # store predictions and real values
        predictions.append(pred.item())
        true_val.append(label.item())
        
        
    
    testing_set_predictions = pd.concat([
        pd.DataFrame(true_val,columns=['BC_real_scaled'],index = ds.test_measurement_date),
        pd.DataFrame(descaler.inverse_transform(pd.DataFrame(true_val)),columns = ['BC_real'],index = ds.test_measurement_date),
        pd.DataFrame(predictions,columns=['BC_predicted_scaled'],index = ds.test_measurement_date),
        pd.DataFrame(descaler.inverse_transform(pd.DataFrame(predictions)),columns = ['BC_predicted'],index = ds.test_measurement_date)],
        axis=1)
    print(f'Testing set predictions\n{testing_set_predictions.head()}')
    print('Real measurements scoring')
    RMSE,R2 = scoringMetrics(testing_set_predictions.loc[:,'BC_real'],testing_set_predictions.loc[:,'BC_predicted'])
    print('Scaled measurements scoring')
    scoringMetrics(testing_set_predictions.loc[:,'BC_real_scaled'],testing_set_predictions.loc[:,'BC_predicted_scaled'])
        
    return testing_set_predictions, RMSE, R2

def scoringMetrics(Y,y_pred):
    """
    Computing scoring metrics using Y and model predictions
    """
    # scoring metrics
    RMSE = np.sqrt(mean_squared_error(Y,y_pred))
    R2 = r2_score(Y,y_pred)
    
    print(f'BC proxy model predictions\n RMSE = {RMSE}\t R2 = {R2}\n')
    
    return RMSE,R2

   
def saveModelPerformance(model,df,RMSE,R2,path,args):
    # save predictions and real values
    fname = 'BC_prediction_values_'+args.architecture+'_'+str(args.hiddenlayers)+'HL'+'_'+str(args.nodeshl)+'nhl'+'_'+str(args.seqlength)+'seqL'+str(args.forecastHours)+'hForecast.csv'
    df.to_csv(path+'/'+fname)
    # save scoring metrics
    fname = 'BC_scoring_'+args.architecture+'_'+str(args.hiddenlayers)+'HL'+'_'+str(args.nodeshl)+'nhl'+'_'+str(args.seqlength)+'seqL_'+str(args.dropoutprob)+'pDO_'+str(args.forecastHours)+'hForecast.txt'
    f = open(path+'/'+fname,'a')
    print(15*'-',file=f)
    print(f'BC model:\n {model}',file=f)
    print(f'Testing scoring\nRMSE = {RMSE}\nR2 = {R2}',file=f)
    f.close()
    return


#%%
    
def main(model,data_module,args):
    # Traininig
    
    if args.train:
        os.chdir(training_path)
        data_module.setup(stage='fit')
        fname_logger = f'{args.architecture}_{args.npredictors}predictors_{args.hiddenlayers}HL_\
{args.nodeshl}nodesHL_{args.dropoutprob}pDO_{args.learningrate}lr_{args.weightDecay}wd\
_{args.epochs}epochs_{args.batchsize}Nbatch_{args.seqlength}SeqL_{args.forecastHours}hForecast'

        fname_ckpt = f'{args.architecture}_{args.npredictors}predictors_{args.hiddenlayers}HL_\
{args.nodeshl}nodesHL_{args.dropoutprob}pDO_{args.learningrate}lr_{args.weightDecay}wd\
_{args.epochs}epochs_{args.batchsize}Nbatch_{args.seqlength}SeqL_{args.forecastHours}hForecast'
            
        logger = TensorBoardLogger(
            save_dir='lightning_loss',
            name=fname_logger,
            log_graph=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath='checkpoints',
            filename=fname_ckpt+'_{epoch:02d}_{val_loss:.2f}_{train_loss:.2f}_bestCheckpoint',
            save_top_k=1,
            verbose=True,
            monitor='val_loss',
            mode='min'
            )
        early_stopping_callback = EarlyStopping(monitor='val_loss',patience=10)
        prog_bar_callback = RichProgressBar()
        profiler = AdvancedProfiler(filename='perf_logs')
        trainer_BC = pl.Trainer(
            logger =logger,
            callbacks = [early_stopping_callback,checkpoint_callback,prog_bar_callback],
            gradient_clip_val=0,
            gradient_clip_algorithm = 'norm',
            max_epochs=args.epochs,
            enable_progress_bar=True,
            accelerator = args.accelerator,
            devices=args.ndevices,
            auto_select_gpus=True,
            log_every_n_steps=5,
            precision=16,
            profiler=profiler,
            # strategy='ddp' #DDPStrategy(find_unused_parameters=False)
            # replace_sampler_ddp=False
            )

        start_time = time.time()
        trainer_BC.fit(model,data_module)
        model_fitting_time = time.time() - start_time
        print('Finished in %.2f s / %.2f min' %(model_fitting_time,model_fitting_time/60))
        return 
    
    # Testing
    else:
        fname_model_ckpt = f'{args.architecture}_{args.npredictors}predictors_{args.hiddenlayers}HL_\
{args.nodeshl}nodesHL_{args.dropoutprob}pDO_{args.learningrate}lr_{args.weightDecay}wd\
_{args.epochs}epochs_{args.batchsize}Nbatch_{args.seqlength}SeqL_{args.forecastHours}hForecast_epoch={args.earlyStoppingEpoch}_val_loss={args.validationLoss}_train_loss={args.trainLoss}_bestCheckpoint.ckpt'
        
        print(f'Loading trained model {fname_model_ckpt} from checkpoints {trained_model_path} ')
        
        trained_model = model.load_from_checkpoint(
            trained_model_path+fname_model_ckpt,
            architecture=args.architecture
            )
        
        print(f'Loaded model:\n{trained_model}')
        
        # test model
        data_module.setup(stage='predict')
        trained_model.eval()
        
        print('Predicting on testing set')
        prog_bar_callback = RichProgressBar()
        trainer = pl.Trainer(
            callbacks = [prog_bar_callback],
            gradient_clip_val=0,
            gradient_clip_algorithm = 'norm',
            max_epochs=args.epochs,
            enable_progress_bar=True,
            accelerator = args.accelerator,
            devices=1,
            auto_select_gpus=True,
            log_every_n_steps=5,
            precision=16
            )

        trainer.test(trained_model, data_module,verbose=True)
        
        testSet_predictions, RMSE, R2 = modelPredictions(args,ds,data_module,trained_model)
        saveModelPerformance(trained_model,testSet_predictions,RMSE,R2,results_path,args)
        
        return testSet_predictions, RMSE, R2
   

    
#%%
if __name__ == '__main__':
    args = script_params()
    abs_path = os.path.dirname(os.path.realpath(__file__))
    # paths for training
    file_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Files'
    training_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Results/'+args.architecture+'/Loss'
    # trained model paths
    trained_model_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Results/'+args.architecture+'/Loss/checkpoints/'
    results_path= os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Results/'+args.architecture+'/predictions'
    
    os.chdir(abs_path)
    
    pl.seed_everything(92, workers=True)
    
    # # variables
    # ## data file parameters
    # sensor_device = args.device
    # data_source = args.datasource
    # fname_proxy_dataset = args.proxydataset
    
    # ## data set parameters
    
    
    # proxy_predictors = args.predictors[0]
    # n_months_traininig= args.monthsTrain
    # n_months_val = args.monthsVal
    # sequence_length = args.seqlength
    # batch_size = args.batchsize
    # scaler = args.scaler
    
    # ## model parameters
    # architecture = args.architecture
    # n_predictors = len(args.predictors[0])-1
    # n_output = args.noutput
    # n_hl_stacked = args.hiddenlayers
    # n_nodes_hl = args.nodeshl
    # N_EPOCHS=args.epochs
    # L_RATE=args.learningrate
    # dropOut_prob = args.dropoutprob
    
    # acc = args.accelerator
    # n_devices = args.ndevices
    

    # data set
    ds = prepareDataSet(args,file_path=file_path,file_name=args.proxydataset)
    data_module = BC_DataModule(ds,args)
    
    
    # model creation
    model = BC_Predictor(args)
    print(f'ANN created\n {model}')
    
    # model training/testing
    main(model=model,data_module=data_module,args=args)
    
    
    
    



