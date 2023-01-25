#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load Raw data sets

Created on Thu Nov 10 16:28:53 2022
@author: jparedes
"""
import os
import pandas as pd
import numpy as np
import warnings
import pickle
from pandas.tseries.offsets import DateOffset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
#%%
class dataSet_raw_refStation():
    def __init__(self,fname):
        if fname not in ['PR_Data2018.xlsx','PR_Data2019.xlsx','Joint']:
            raise Exception('Enter a valid name')
        print(f'Preparing data set: {fname}\nFor loading the data set use load_ds(time_period=Xmin)')
        self.fname = fname
        self.idx = [0,2,4,3,5,11,9]# idx of sheets used
        self.df = pd.DataFrame()
        if self.fname == 'Joint':
            print(f'{self.fname} data set specified. Concatenate previous data sets using Merge_datasets()')
    
    def load_ds(self,time_period='60min'):
        if self.fname == 'Joint':
            raise Exception('Joint data set is the composition of 2 or more previously loaded data sets.\nFirst load a proper data set')
        for i in self.idx:
            df = pd.read_excel(self.fname,sheet_name=i)
            df.rename(columns={df.columns[0]:'date'},inplace=True)
            df.date = pd.to_datetime(df.date)
            if i == 9:# keep only certain entries from Meteo data
                df = df.loc[:,['date','TEMP','HUM','PRES','Vmax']]
            
            #specific adjustments
            if self.fname == 'PR_Data2018.xlsx':
                if i == 2:
                    df.date = df.date.dt.round(freq='5T')
                elif i in [3,4,5]:   
                    df.date.dt.round(freq='10T')
            elif self.fname == 'PR_Data2019.xlsx':
                if i in [2,4]:
                    df.date = df.date.dt.round(freq='5T')
                elif i in [5,6]:
                    df.date = df.date.dt.round(freq='10T')
                elif i == 11:
                    df.date = df.date.dt.round(freq='1H')
            
            # correct wrong values
            df.iloc[:,1:] = np.abs(df.iloc[:,1:])# replace with abs val
            #df.iloc[(df.iloc[:,1]<0.0).values,1]=0.0# replace with zero
            #df.iloc[(df.iloc[:,1]<0.0).values,1]=np.nan# replace with NaN
            
            
                    
                    
                
            df_ = df.groupby(pd.Grouper(key='date',freq=time_period)).mean()
            self.df = pd.concat([self.df,df_],axis=1)
        
        # Dropping missing values
        #self.df.dropna(inplace=True)    
    
    def Merge_datasets(self,*args):
        if self.fname != 'Joint':
            raise Exception(f'Concatenation operation requires different data sets.\nsingle data set loaded {self.fname}.  Must create a Joint data set.')
        if args == ():
            raise Exception('No data sets specified for concatenation')
        if self.df.shape[0] != 0:
            warnings.warn('Data set not empty. New data frames will be concatenated to previous data set')
            
        print('Concatenating data sets')
        for ds in args:
            self.df = pd.concat([self.df,ds],axis=0)
        
        # Specify simpler names
        var = ['BC','N','PM1','PM25','PM10','SO2','NO','NO2','O3','CO','NOx','T','RH','P','Vmax']
        self.df.columns = var
        
        # correct units
        self.df.loc[:,'BC'] = 1e-3*self.df.loc[:,'BC'] #ng to ug 
        self.df.loc[:,'N'] = 1e6*self.df.loc[:,'N'] # n/cm3 to n/m3
        
class dataSet():
    def __init__(self,files_path,device='Ref_st',source='raw'):
        if device not in ['Ref_st','LCS']:
            raise Exception('Incorrect deivce. Choose either Ref_st or LCS')
        if source not in ['raw','data_frame']:
            raise Exception('Incorrect data source. Choose either raw or data_frame')
        
        print('Initializing data set')
        self.device = device
        self.source = source
        self.files_path = files_path
        self.df = pd.DataFrame()
        print(f'Prepared to load {self.device} data set from {self.source}')
    
    def load_raw(self):
        print('Loading data set from raw files')
        print(f'Loading data for {self.source}')
        if self.device == 'Ref_st':
            path = os.getcwd()
            path = path + '/Files/Raw_data/Reference_station'
            os.chdir(path)
            print(f'Changing directory to {path}')
            fname = 'PR_Data2018.xlsx'
            ds18 = dataSet_raw_refStation(fname = fname)
            ds18.load_ds()
            fname = 'PR_Data2019.xlsx'
            ds19 = dataSet_raw_refStation(fname = fname)
            ds19.load_ds()
            ds = dataSet_raw_refStation(fname='Joint')
            ds.Merge_datasets(ds18.df,ds19.df)
            self.df = ds.df
        
        elif self.device == 'LCS':        
            print('LCS not implemented yet')
            self.df = pd.DataFrame()
            
    def load_dataFrame(self,time_freq='60min'):
        if self.device == 'Ref_st':
            path = self.files_path + '/dataFrames/Reference_station/'
            print(f'Loading dataFrame from {path}')
            fname = 'RefStation_'+time_freq+'.pkl'
            with open(path+fname,'rb') as f:
                self.df = pickle.load(f)
                
        elif self.device == 'LCS':
            print('LCS not implemented yet')
            self.df = pd.DataFrame()
            
    def load_dataSet(self):
        if self.source == 'raw':
            self.load_raw()
            
        elif self.source == 'data_frame':
            self.load_dataFrame()
        
#%% Tr/Val/Ts sets
class dataSet_split():
    
    def __init__(self,df,train=12,val=6):
        if not isinstance(train,int):
            raise TypeError('Training size must be an integer')
        if not isinstance(val,int):
            raise TypeError('Validation size must be an integer')
            
        self.df = df
        self.n_samples = df.shape[0]
        self.train = train
        self.val = val
        
        self.train_set = pd.DataFrame()
        self.val_set = pd.DataFrame()
        self.test_set = pd.DataFrame()
        print(f'Number of samples in data set: {self.n_samples}')
        print(f'{self.train} months for training\n{self.val} months for validation')
        
        
    def train_val_test_split(self):
        # assign training set
        offSet = self.df.index[0]+DateOffset(months=self.train)
        date_range = pd.date_range(start=self.df.index[0],end=offSet,freq='1H',closed='left')
        x=self.df.index[self.df.index.isin(date_range)]
        self.train_set = self.df.loc[x]
        # assign validation set
        start = x[-1]
        if not start.is_month_end:
            start += pd.tseries.offsets.MonthBegin()
            start = start.floor('D')
        else:
            start = start.ceil('D')
        end = start+DateOffset(months=self.val)
        date_range = pd.date_range(start=start,end=end,freq='1H',closed='left')
        x = self.df.index[self.df.index.isin(date_range)]
        self.val_set = self.df.loc[x]
        # assign testing set
        start = x[-1].ceil('D')
        end = self.df.index[-1]
        date_range = pd.date_range(start=start,end=end,freq='1H')
        x = self.df.index[self.df.index.isin(date_range)]
        self.test_set = self.df.loc[x]
        
    def countnan(self):
        # count NaNs
        self.nan_full = self.df.isna().sum()
        self.nan_trainingSet = self.train_set.isna().sum()
        self.nan_valSet = self.val_set.isna().sum()
        self.nan_testingSet = self.test_set.isna().sum()
        # valid measurements
        self.valid_full = self.df.notna().sum()
        self.valid_trainingSet = self.train_set.notna().sum()
        self.valid_valSet = self.val_set.notna().sum()
        self.valid_testingSet = self.test_set.notna().sum()
        # report
        print(f'Columns on data set:\n{self.df.columns}')
        print(f'Full data set missing values\n{self.nan_full}')
        print(f'Full data set valid measurements\n{self.valid_full}')
        print('-----')
        print(f'Training set missing values\n{self.nan_trainingSet}')
        print(f'Training set valid measurements\n{self.valid_trainingSet}')
        print('-----')
        print(f'Validation set missing values\n{self.nan_valSet}')
        print(f'Validation set valid measurements\n{self.valid_valSet}')
        print('-----')
        print(f'Testing set missing values\n{self.nan_testingSet}')
        print(f'Testing set valid measurements\n{self.valid_testingSet}')
        print('-----')
            
    def Scale(self,scaler='Standard'):
        if not isinstance(scaler,str):
            raise TypeError('scaler must be a string')
        if scaler not in ['Standard','MinMax']:
            raise Exception('Incorrect scaler. Choose between Standard or MinMax')
        
        # normalize data set
        if scaler == 'Standard':
            self.scaler = StandardScaler()
        elif scaler == 'MinMax':
            self.scaler = MinMaxScaler()
        self.scaler.fit(self.train_set)
        self.train_set_scaled = pd.DataFrame(self.scaler.transform(self.train_set),
                                             index=self.train_set.index,
                                             columns=self.train_set.columns)
        self.val_set_scaled = pd.DataFrame(self.scaler.transform(self.val_set),
                                           index = self.val_set.index,
                                           columns=self.val_set.columns)
        self.test_set_scaled = pd.DataFrame(self.scaler.transform(self.test_set),
                                            index = self.test_set.index,
                                            columns = self.test_set.columns)
        
    def create_sequences(self,input_set = 'Train', target_column='BC', sequence_length=2,auto_regressive=False, forecasting=True):
        if target_column != 'BC':
             warnings.warn('The target pollutant is not Black carbon (BC)!')
        if not isinstance(sequence_length,int):
            raise TypeError('The sequence length must be an integer')
        if input_set not in ['Train','Validation','Test']:
            raise Exception('input_data must be either Train, Validation or Test')
            
        if input_set == 'Train':
            input_data = self.train_set_scaled
        elif input_set == 'Validation':
            input_data = self.val_set_scaled
        elif input_set == 'Test':
            input_data = self.test_set_scaled
            
        print(f'Creating forecasting set on {input_set} for {target_column} using last {sequence_length} measurements')
        sequences = []
        data_size = input_data.shape[0]
        for i in tqdm(range(data_size-sequence_length)):
            
            if auto_regressive:
                sequence = input_data.iloc[i:i+sequence_length,0:]
            else:
                sequence = input_data.iloc[i:i+sequence_length,1:]
            
            if forecasting:
                # save as label the value at nex time step
                label_position = i+sequence_length
                label = input_data.iloc[label_position][target_column]
            else:
                # not forecasting. save the value at the same time step
                label_position = i
                label = input_data.iloc[label_position][target_column]
            sequences.append((sequence,label))
        
        if input_set == 'Train':
            self.train_set_sequence = sequences
        elif input_set == 'Validation':
            self.val_set_sequence = sequences
        elif input_set == 'Test':
            self.test_set_sequence = sequences
        self.sequence_length = sequence_length
        
        
    def create_sequences_dates(self,input_set = 'Train', target_column='BC', sequence_length=2,n_hours_forecast=0,auto_regressive=False):
        """
        Creates time sequence using the last {sequence_length} hours with mneasurements to predict the {target_column}in the data set
        with {n_hours_forecast} hours in the future
        """
        if target_column != 'BC':
             warnings.warn('The target pollutant is not Black carbon (BC)!')
        if not isinstance(sequence_length,int):
            raise TypeError('The sequence length must be an integer')
        if input_set not in ['Train','Validation','Test']:
            raise Exception('input_data must be either Train, Validation or Test')
            
        if input_set == 'Train':
            input_data = self.train_set_scaled
        elif input_set == 'Validation':
            input_data = self.val_set_scaled
        elif input_set == 'Test':
            input_data = self.test_set_scaled
            
        print(f'Creating sequences on {input_set} set for {target_column} using last {sequence_length} measurements to forecast {n_hours_forecast} hours')
        if auto_regressive:
            print(f'Auto regressive model: Target variable {target_column} will be included as a predictor')
        sequences = []
        data_size = input_data.shape[0]
        measurement_date = []
        # Checkpoint
        for i in tqdm(range(sequence_length+n_hours_forecast - 1,data_size)):
            label_position = i
            label = input_data.iloc[label_position][target_column]
            #end_date = input_data.iloc[label_position].name-DateOffset(hours=1)
            end_date = input_data.iloc[label_position].name - DateOffset(hours=n_hours_forecast)
            start_date = end_date - DateOffset(hours=sequence_length)
            sequence_date = pd.date_range(start=start_date,end=end_date,freq='1H',closed='right')
            if auto_regressive:
                sequence = input_data.loc[input_data.index.isin(sequence_date)].iloc[:,0:]
            else:
                sequence = input_data.loc[input_data.index.isin(sequence_date)].iloc[:,1:]
            
            if sequence.shape[0] == sequence_length:# only save sequences with the exact number of measurements in the time window: no padding 
                sequences.append((sequence,label))
                measurement_date.append(end_date + DateOffset(hours=1))
                
        
        if input_set == 'Train':
            self.train_set_sequence_dates = sequences
            self.train_measurement_date = measurement_date
        elif input_set == 'Validation':
            self.val_set_sequence_dates = sequences
            self.val_measurement_date = measurement_date
        elif input_set == 'Test':
            self.test_set_sequence_dates = sequences
            self.test_measurement_date = measurement_date
        self.sequence_length = sequence_length
        
                
        
        



#%% main

def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))    
    # go to file location
    device = 'Ref_st' # Ref_st or LCS
    source = 'data_frame'   # raw or data_frame
    ds = dataSet(device=device,source=source) 
    ds.load_dataSet()
    df = ds.df
    ds = dataSet_split(df,train=12,val=6)
    ds.train_val_test_split()
    scaler = 'Standard'# Standard or MinMax
    ds.Scale(scaler=scaler)
    # create forecasting time window
    time_window_length = 12
    ds.create_sequences(input_set = 'Train', target_column='BC', sequence_length=time_window_length)
    ds.create_sequences(input_set = 'Validation', target_column='BC', sequence_length=time_window_length)
    ds.create_sequences(input_set = 'Test', target_column='BC', sequence_length=time_window_length)

    return df,ds
#%%
if __name__ == '__main__':
    df,ds = main()
