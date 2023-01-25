#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 11:47:02 2022

@author: jparedes
"""

import time
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt

import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# modules
import Load_dataSet as LDS
import Plots


#%% base functions
def set_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    return device
#%% DataSet
def pandas2tensor(df):
    
    class MyDataset(Dataset):
        def __init__(self):
            y=df.iloc[:,0].values
            x=df.iloc[:,1:].values
    
            self.x_train=torch.tensor(x,dtype=torch.float32)
            self.y_train=torch.tensor(y,dtype=torch.float32)
 
        def __len__(self):
            return len(self.y_train)
   
        def __getitem__(self,idx):
            return self.x_train[idx],self.y_train[idx]
    
    df_tensor = MyDataset()
    return df_tensor
    


#%% NNs
def Load_NN(fname,model_init):
    """
    Loads the parameters of a previous ANN. The network must be created.
    The loaded network is set to evaluation mode

    Parameters
    ----------
    fname : str
        name of file containing network parameters
    model_init : pytorch NeuralNetwork
        initialized ANN

    Returns
    -------
    model_init : pytorch NeuralNetwork
        ANN with updated weights

    """
    model_init.load_state_dict(torch.load(fname))
    model_init.eval()
    return model_init

def NN_create(n_input,n_output,n_nodes_hl,device,p=0.1,seed=92):
    torch.manual_seed(seed)
    
    # Custom layers or activation
    class CreateMissings(nn.Module):
        # works as an activation for specific layer
        def __init__(self,p=0.1):
            super(CreateMissings,self).__init__()
            self.p = p
            #self.mask 
            
            
        def forward(self,L):
            if self.training:
                #mask = (x_previous != 0).all(0).float()
                #x += x* mask
                mask = torch.rand(L.size())>self.p
                L = L*mask
                return L
            
            
    # NN definition
    class NeuralNetwork(nn.Module):
        def __init__(self):#def __init__(self,**args):
            super(NeuralNetwork, self).__init__()
            self.dropoutInput = nn.Dropout(p)
            self.InputNorm = nn.BatchNorm1d(n_input)
            self.HL1 = nn.Linear(n_input,n_nodes_hl,bias=True)
            nn.init.xavier_uniform_(self.HL1.weight, gain=1.0)
            self.OutputLayer = nn.Linear(n_nodes_hl, n_output,bias=True)
            torch.nn.init.xavier_uniform_(self.OutputLayer.weight, gain=1.0)
            
            # sequential: just for testing
            self.mlp = nn.Sequential(
                nn.Dropout(p),
                nn.Identity(),
                )

        def forward(self, x):
            #forward pass
            out = self.dropoutInput(x)
            out = self.InputNorm(out)
            out = nn.ReLU()(self.HL1(out))    
            y_hat = self.OutputLayer(out)
            return y_hat
        
        
    model = NeuralNetwork().to(device)
    print(model)
        
    return model
#-------------------------
# Temporal BC proxy: TBCP
#-------------------------
## RNN TBCP
class TBCP_RNN(nn.Module):
    
    def __init__(self,n_predictors,n_hidden=10, n_layers=1,n_output=1):
        super().__init__()
        self.n_hidden = n_hidden
        
        # time step pass
        self.rnn = nn.RNN(input_size=n_predictors,
                          hidden_size = n_hidden,
                          num_layers = n_layers,
                          nonlinearity = 'relu',
                          bias = True,
                          batch_first = True,
                          dropout = 0.1,
                          bidirectional = False
                          )
        
        # regression for last layer
        self.regressor = nn.Linear(n_hidden,n_output,bias=True)
        
    def forward(self,x):
        # Input.size = (batch_size, sequence_length, n_predictors)
        self.rnn.flatten_parameters()# only CUDA
        
        # time step prediction
        out,_ = self.rnn(x)
        
        # proxy on predictions
        return self.regressor(out)
        
        
        
        

#%% Training
def NN_train_loop(model,dataloader,lr,decay,loss_fn,optimizer):
    size = len(dataloader.dataset)
    
    # training loop
    model = model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred.squeeze(), y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    return loss

def NN_val_loop(dataloader, model,loss_fn):
    model.eval()
    
    num_batches = len(dataloader)
    test_loss = 0.

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred.squeeze(), y).item()

    test_loss /= num_batches
    print(f"Validation Error: \n Avg loss: {test_loss:>8f} \n")
    
    return test_loss

def NN_train(model,train_data,val_data,batch_size=32,n_epochs=4,lr=1e-3,decay=0,opt='SGD',save_network=False):
    """
    

    Parameters
    ----------
    model      : Pytorch NN
        Initialized neural network
    train_data : TYPE
        DESCRIPTION.
    val_data : TYPE
        DESCRIPTION.
    
        NN hyperparams:
        ---------------
        
        batch_size : int
            The number of data samples propagated through the network before the parameters are updated.
            The default is 32
        
        n_epochs : int
            The number times to iterate over the dataset.
            The default is 10.
         
        lr : float
            Rate to update models parameters at each batch/epoch. 
            Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.
            The default is 1e-3.
        
        decay : float
            Weight decay regularization. The default is 0 (no regularization).

    Returns
    -------
    None.

    """
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    
    loss_fn = nn.MSELoss(reduction='mean')
    if opt =='SGD':
        optimizer = torch.optim.SGD(model.parameters(),lr=lr,weight_decay=decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.999),eps=1e-8,weight_decay=decay)
    
    # Training
    Loss_train = []
    Loss_test = []
    for t in range(n_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss_train = NN_train_loop(model,train_dataloader,lr,decay,loss_fn,optimizer)
        loss_test = NN_val_loop(test_dataloader, model, loss_fn)
        Loss_train.append(loss_train)
        Loss_test.append(loss_test)
    
    if save_network:
        fname = 'model.pth'
        torch.save(model, 'model.pth')  
        print(f'Model saved as: {fname}')
        fname = 'model.params'
        torch.save(model.state_dict(), fname)
        print(f'Model parameters saved as: {fname}')
        
        
    return Loss_train,Loss_test
    
#%%
def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # load data set
    device = 'Ref_st' # Ref_st or LCS
    source = 'data_frame'   # raw or data_frame
    ds = LDS.dataSet(device=device,source=source) 
    ds.load_dataSet()
    df = ds.df
    ds = LDS.dataSet_split(df,train=12,val=6)
    ds.train_val_test_split()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    
    # create NN architecture
    pl.seed_everything(92, workers=True)
    ## network parameters
    device = set_device()
    n_input = 5
    n_output = 1
    n_nodes_hl = 5
    input_dropOut_prob = 0.1
    weights_seed = 92

    ## model creation
    model = NN_create(n_input,n_output,n_nodes_hl,device,p=input_dropOut_prob,seed=weights_seed)
    print(f'ANN created\n {model}')
    print(f'Network parameters\n {model.state_dict()}')
    return df,ds,model

if __name__ == '__main__':
    df,ds,model = main()


