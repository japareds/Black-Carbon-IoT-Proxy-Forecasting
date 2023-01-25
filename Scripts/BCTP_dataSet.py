#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loads raw measrement files and create .pkl data frames ready to use

Created on Mon Oct 17 15:12:47 2022

@author: jparedes
"""


import os
import pandas as pd
#%%
### load pkl
def Load_pkl_datasets(path,pollutant,fname):
    """
    Basic function for loading air pollution .pkl data sets 
    and return pandas data frame
    
    Parameters
    ----------
    path: str
            file path
    pollutant: str
            pollutant name
    fname: str
            file name of pollutant file
    
    Returns
    -------
    df: pandas DataFrame
        pandas data frame with the corresponing data set
    
    """
    os.chdir(path)
    ### target pollutant
    print('Loading '+ pollutant +' data set from: '+ path +'\n')
    df = pd.read_pickle(fname)
    return df

### load LCS data sets

def Load_cal_pollutants(path,cal_pollutants,extension='_cal_SVR_calibrated_SVR.pkl'):
    """
    path: file path
    cal_pollutant: array with str name of pollutants
    extension: all calibrated pollutants differ only on the first part (name)
    """
    df_cal = []
    for pollutant in cal_pollutants:
            print('Loading SVR calibrated ',pollutant)
            fname = pollutant+extension
            df_cal.append(Load_pkl_datasets(path,pollutant,fname))
            
    df_O3, df_NO2, df_NO, df_PM10 = df_cal[0],df_cal[1],df_cal[2],df_cal[3]
    return df_O3,df_NO2,df_NO,df_PM10

def Load_NonCal_predictors(path,n_sensor=4):
    """
    path: path to file
    n_sensor: from 8 possible sensors, number of the one that is not dropped
    """
    os.chdir(path)
    print('Loading non-calibrated LCS measurements')
    LCS_PM = Load_pkl_datasets(path,'PM','PM_JJ.pkl')
    LCS_PM25 = Load_pkl_datasets(path,'PM','PM25_JJ.pkl')
    LCS_N = Load_pkl_datasets(path,'PM','N_JJ.pkl')
    ### keep only 1 sensor
    LCS_PM25 = LCS_PM25[n_sensor]
    LCS_PM = LCS_PM[n_sensor]
    LCS_N = LCS_N[n_sensor]
    ### One entry of LCS_PM is PM10 and the other is PM1
    LCS_PM1 = LCS_PM.iloc[:,0:2]
    LCS_PM10 = LCS_PM.iloc[:,0:3:2]
    return LCS_PM1,LCS_PM25,LCS_PM10,LCS_N

### load Reference Station data set
def Load_RefSt_predictors(path,*fname):
    """
    loads predictors from Ref. Station dataset
    For calibration: O3, NO2, NO, PM10
    others. Meteorological predictors: T and Rh
    
    Parameters
    ----------    
    path: str
        file path
        
    *fname: tuple of str
        file names
    """
    Ref_O3 = Load_pkl_datasets(path,'Reference O3',fname[0])
    try:
        Ref_NO2 = Load_pkl_datasets(path,'Reference NO2 ', fname[1])
        Ref_NO = Load_pkl_datasets(path,'Reference NO ', fname[2])
        Ref_PM10 = Load_pkl_datasets(path,'Reference PM 10',fname[3])
        meteo = Load_pkl_datasets(path,'meteorologial',fname[-1])
    except Exception as e:
        print('Unable to load more than 1 pollutant')
        print(e)
    return Ref_O3,Ref_NO2,Ref_NO,Ref_PM10,meteo



#%% load raw data sets
### load csv
def load_captor_csv(Captor_path):
    """
    Load O3, NO2, NO data sets from csv file
    and return pandas DataFrame

    Parameters
    ----------
    Captor_path : str
            path to csv files

    Returns
    -------
    LCS_O3 : pandas DataFrame
        data frame containing O3 raw data set
    LCS_NO2 : pandas DataFrame
        data frame containing NO2 raw data set
    LCS_NO : pandas DataFrame
        data frame containing NO raw data set
    LCS_Meteo : pandas DataFrame
        data frame containig inner T and RH raw data set

    """
    os.chdir(Captor_path)
    ### months of data
    months = ["October","November","December"]
    first = True
    for m in months:
        path = Captor_path +'/' +m + "/"
        os.chdir(path)
        # loading data 50/52/T
        file_name = "def_data50.csv"
        print('Loading Captor data set %s from %s'%(file_name,path))
        data50 = pd.read_csv(file_name, sep=";", header=0, names=['id', 'N1', 'N2', 'WE_NO', 'AE_NO', 'N3', 'N4', 'date'])
        file_name = "def_data52.csv"
        print('Loading Captor data set %s from %s'%(file_name,path))
        data52 = pd.read_csv(file_name, sep=";", header=0, names=['id', 'WE_O3', 'AE_O3', 'WE_NO2', 'AE_NO2', 'N3', 'N4', 'date'])
        file_name = "def_dataT-H.csv"
        print('Loading Captor data set %s from %s'%(file_name,path))
        dataT = pd.read_csv(file_name, sep=";", header=0,names=['id', 'T', 'RH', 'date'])
    
        if first:
            df50 = data50.copy()
            df52 = data52.copy()
            dfT = dataT.copy()
            first = False
        else:
            df50 = pd.concat([df50,data50])
            df52 = pd.concat([df52,data52])
            dfT = pd.concat([dfT,dataT])
   
    ### format date
    df50.date = pd.to_datetime(df50.date)
    df52.date = pd.to_datetime(df52.date)
    dfT.date = pd.to_datetime(dfT.date)
    ### create LCS data sets
    LCS_NO = df50.loc[:,['date','WE_NO','AE_NO']].copy()
    LCS_NO2 = df52.loc[:,['date','WE_NO2','AE_NO2']].copy()
    LCS_O3 = df52.loc[:,['date','WE_O3','AE_O3']].copy()
    ### Add column with electrodes readings difference
    LCS_NO['S_NO'] = LCS_NO.loc[:,'WE_NO']-LCS_NO.loc[:,'AE_NO'] #LCS_NO.apply(lambda row: row.WE_NO - row.AE_NO, axis=1)
    LCS_NO2['S_NO2'] = LCS_NO2.loc[:,'WE_NO2']-LCS_NO2.loc[:,'AE_NO2'] #LCS_NO2.apply(lambda row: row.WE_NO2 - row.AE_NO2,axis=1)
    LCS_O3['S_O3'] = (LCS_O3.loc[:,'WE_O3']-LCS_O3.loc[:,'AE_O3'])-LCS_NO2.loc[:,'S_NO2'] #LCS_O3.apply(lambda row: (row.WE_O3 - row.AE_O3)-LCS_NO2.S_NO2, axis=1)

    ### Meteorological values from Captor data is the internal reading, not environmental
    LCS_Meteo = dfT.loc[:,['date','T','RH']].copy()
    LCS_Meteo.columns=['date','T_int','RH_int']
   
    return LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo

def load_PM_csv(PM_path,sensor_i=5):
    """
    load PM sensor csv files and creates pandas Data Frame
    with the data set

    Parameters
    ----------
    PM_path : str
        path to csv file
        
    sensor_i : int, optional. Default value 5
        sensor numbering to load. must be between [1,8]

    Returns
    -------
    LCS_PM1: pandas DataFrame
        data frame containing raw PM1 data set
    LCS_PM25: pandas DataFrame
        data frame containing raw PM25 data set
    LCS_PM10: pandas DataFrame
        data frame containing raw PM10 data set
    LCS_N: pandas DataFrame
        data frame containig raw UFP data set. 
        Multiple columns for different threshold sizes

    """
    os.chdir(PM_path)
    
    
    ### 8 possible sensors, load 1 of them
    #  don't concat the datasets! 
    #  primary and secondary datasets have measurements at different seconds
    #  different sensors have measurements at different minutes

    PM_A = 'JJ'+str(sensor_i)+'_A.csv'
    PM_S = 'JJ'+str(sensor_i)+'_AS.csv'
    print('Loading PM and UFP data set .csv files %s\n %sfrom %s'%(PM_A,PM_S,PM_path))
    
    df_A = pd.read_csv(PM_A)
    df_AS = pd.read_csv(PM_S)
    
    ### variables of interest from primary and secondary datasets
    LCS_PM1= df_AS.loc[:,['created_at','PM1.0_ATM_ug/m3']].copy()
    LCS_PM25 = df_A.loc[:,['created_at','PM2.5_ATM_ug/m3']].copy()
    LCS_PM10 = df_AS.loc[:,['created_at','PM10_ATM_ug/m3']].copy()
    LCS_N = df_AS.loc[:,['created_at','>=0.3um/dl','>=0.5um/dl','>1.0um/dl','>=2.5um/dl',\
                        '>=5.0um/dl','>=10.0um/dl']].copy()
    
    
    ### rename columns: include sensor number
    LCS_PM1.columns = ['date','S'+str(sensor_i)+'_PM1']
    LCS_PM25.columns = ['date','S'+str(sensor_i)+'_PM25']   
    LCS_PM10.columns = ['date','S'+str(sensor_i)+'_PM10']
    LCS_N.columns = ['date','S'+str(sensor_i)+'_0.3','S'+str(sensor_i)+'_0.5','S'+str(sensor_i)+'_1.0','S'+str(sensor_i)+'_2.5',\
            'S'+str(sensor_i)+'_5.0','S'+str(sensor_i)+'_10.0']
    
    ### datetime format
    LCS_PM1['date'] = LCS_PM1['date'].str.replace('UTC', '')
    LCS_PM25['date'] = LCS_PM25['date'].str.replace('UTC', '')
    LCS_PM10['date'] = LCS_PM10['date'].str.replace('UTC', '')
    LCS_N['date'] = LCS_N['date'].str.replace('UTC', '')
    
    LCS_PM1.date = pd.to_datetime(LCS_PM1.date,format="%Y-%m-%d %H:%M:%S")
    LCS_PM25.date = pd.to_datetime(LCS_PM25.date,format="%Y-%m-%d %H:%M:%S")
    LCS_PM10.date = pd.to_datetime(LCS_PM10.date,format="%Y-%m-%d %H:%M:%S")
    LCS_N.date = pd.to_datetime(LCS_N.date,format="%Y-%m-%d %H:%M:%S")
    
    
    return LCS_PM1,LCS_PM25,LCS_PM10,LCS_N

def load_Ref_csv(path):
    """
    Load reference station reference measurements from path
    and create pandas DataFrame

    Parameters
    ----------
    path : str
        path to csv file

    Returns
    -------
    Ref_O3 : pandas DataFrame
            data frame containing reference station O3 measurements
    
    Ref_NO2 : pandas DataFrame
            data frame containing reference station NO2 measurements
    
    Ref_NO : pandas DataFrame
            data frame containing reference station NO measurements
    
    Ref_PM10 : pandas DataFrame
            data frame containing reference station PM10 measurements
    
    Ref_Meteo: pandas DataFrame
            data frame containing reference station T and RH measurements

    """
    
    ### read csv and create data frames
    ### gaseous pollutants
    os.chdir(path)   
    file_name = 'Gases_PR_10min_resolution.xlsx'
    print('Loading O3, NO2, NO and PM10 Reference Station dataset file %s from %s'%(file_name,path))
    
    
    Ref_O3 = pd.read_excel(file_name,header=0,usecols=(0,4))
    Ref_O3.columns = ['date','O3_Ref']
    Ref_O3.date = pd.to_datetime(Ref_O3.date)  
    Ref_NO2 = pd.read_excel(file_name,header=0,usecols=(0,3))
    Ref_NO2.columns = ['date','NO2_Ref']
    Ref_NO2.date = pd.to_datetime(Ref_NO2.date)
    Ref_NO= pd.read_excel(file_name,header=0,usecols=(0,2))
    Ref_NO.columns = ['date','NO_Ref']
    Ref_NO.date = pd.to_datetime(Ref_NO.date)
    ### PM10
    file_name = 'Gases_PR_10min_resolution.xlsx'
    Ref_PM10 = pd.read_excel(file_name,header=0,usecols=(0,5))
    Ref_PM10.columns = ['date','PM10_Ref']
    Ref_PM10.date = pd.to_datetime(Ref_PM10.date)
    ### UFP
    file_name = 'N_PR_August21-May22.xlsx'
    print('Loading UFP Reference Station dataset file %s from %s'%(file_name,path))
    
    Ref_N = pd.read_excel(file_name,header=0,usecols=(0,1))
    Ref_N.columns = ['date','N_Ref']
    Ref_N.date = pd.to_datetime(Ref_N.date)
    ### read meterological data
    file_name = 'Meteo-data_SD2021.dat'
    print('Loading Meteorological Reference Station dataset file %s from %s'%(file_name,path))
    
    Ref_Meteo = pd.read_csv(file_name,delimiter=' ')
    Ref_Meteo  = Ref_Meteo [['DATA','HORA','TEMP','HUM']]
    Ref_Meteo = Ref_Meteo.iloc[1:,:]
    ### format meteo columns
    Ref_Meteo['date'] = Ref_Meteo[['DATA', 'HORA']].agg(' '.join, axis=1)
    Ref_Meteo.date = pd.to_datetime(Ref_Meteo.date,format='%Y-%m-%d %H:%M:%S')
    Ref_Meteo = Ref_Meteo[['date','TEMP','HUM']]
    Ref_Meteo.columns = ['date','T','RH']
    
    Ref_Meteo.iloc[:,1] = pd.to_numeric(Ref_Meteo.iloc[:,1], errors='coerce')
    Ref_Meteo.iloc[:,2] = pd.to_numeric(Ref_Meteo.iloc[:,2], errors='coerce')
    
    
    return Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo

def load_BC_csv(path):
    """
    Load black carbon csv data set from path

    Parameters
    ----------
    path : str
        path to BC file

    Returns
    -------
    Ref_BC = pandas DataFrame
            data frame containing black carbon refernce station data set

    """
    os.chdir(path)
    BC_file = 'BC_PR_20-21-22.xlsx'
    print('Loading Black carbon file: %s from %s'%(BC_file,path))
    Ref_BC = pd.read_excel(BC_file)
    ### datetime format
    Ref_BC.columns = ['date','BC']
    Ref_BC.date = pd.to_datetime(Ref_BC.date)
    ### nanograms to micrograms
    Ref_BC.BC = Ref_BC.BC/1000 
    return Ref_BC

#%%
"""
Load data sets and aggregate them
"""
def Load_Pollutants_csv(path):
    """
    
    Reads .csv files from path and returns 
    pandas DataFrame for every pollutant
    
    Parameters
    ----------
    path: str
        path to files. Ideally all are in the same folder
        
    Returns
    -------
    Ref_BC : pandas DataFrame
        Reference Station Black carbon data set
        
    Ref_O3: pandas DataFrame
        Reference Station O3 data set
        
    Ref_NO2: pandas DataFrame
        Reference Station NO2 data set
        
    Ref_NO: pandas DataFrame
        Reference Station NO data set
    
    Ref_PM10: pandas DataFrame
        Reference Station PM10 data set
        
    Ref_Meteo : pandas DataFrame
        Reference Station Meteorological data set
                
    LCS_O3: pandas DataFrame
            LCS O3 data set
            
    LCS_NO2: pandas DataFrame
            LCS NO2 data frame
            
    LCS_NO: pandas DataFrame
            LCS NO data set
        
    LCS_Meteo: pandas DataFrame
            LCS inner device meteorological data set
            
    LCS_PM1 : pandas DataFrame
            LCS PM1 data set
            
    LCS_PM25 : pandas DataFrame
            LCS PM25 data set
            
    LCS_PM10: pandas DataFrame
            LCS PM10 data set
            
    LCS_N : pandas DataFrame
            LCS UFP data set
            
    O3_cal : pandas DataFrame
        Calibrated LCS O3 data set
        
    NO2_cal : pandas DataFrame
        Calibrated LCS NO2 data set
        
    NO_cal : pandas DataFrame
        Calibrated LCS NO data set
        
    PM10_cal : pandas DataFrame
        Calibrated LCS PM10 data set

    """
    print('Loading air pollution csv files')
    Ref_path = path+'/Reference_Station'
    LCS_PM_path = path+'/LCS/PM'
    LCS_Captor_path = path+'/LCS/Captor'
    ### BC
    Ref_BC = load_BC_csv(Ref_path)
    ### Reference Station predictors
    Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo = load_Ref_csv(Ref_path)
    ### LCS predictors
    LCS_PM1,LCS_PM25,LCS_PM10,LCS_N = load_PM_csv(LCS_PM_path)
    LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo = load_captor_csv(LCS_Captor_path)
    
    
    
    return Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N

### save the loaded dataFrames
def save_df(path,Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N):
    """
    Save the created dataFrames 
    with function Load_Pollutants_csv()
    to .pkl in path

    Parameters
    ----------
    path : str
        directory where .pkl files will be stores
    
    Returns
    -------
    None.

    """
    
    path = path+'/dataFrames_files'
    os.chdir(path)
    print('Saving DataFrames to %s'%path)
    
    ### Reference Station data sets
    Ref_BC.to_pickle('Ref_BC.pkl')
    Ref_O3.to_pickle('Ref_O3.pkl')
    Ref_NO2.to_pickle('Ref_NO2.pkl')
    Ref_NO.to_pickle('Ref_NO.pkl')
    Ref_PM10.to_pickle('Ref_PM10.pkl')
    Ref_N.to_pickle('Ref_N.pkl')
    Ref_Meteo.to_pickle('Ref_Meteo.pkl')
    ### LCS data sets
    LCS_O3.to_pickle('LCS_O3.pkl')
    LCS_NO2.to_pickle('LCS_NO2.pkl')
    LCS_NO.to_pickle('LCS_NO.pkl')
    LCS_Meteo.to_pickle('LCS_Meteo.pkl')
    
    LCS_PM1.to_pickle('LCS_PM1.pkl')
    LCS_PM25.to_pickle('LCS_PM25.pkl')
    LCS_PM10.to_pickle('LCS_PM10.pkl')
    LCS_N.to_pickle('LCS_N.pkl')
    
    print('All data frames saved')
    
    return None

def load_pkl(path,pollutant):
    """
    Basic function for loading air pollution .pkl 
    data sets and return pandas data frame
    
    Parameters
    ----------
    path: str
        file path
        
    pollutant: str
        pollutant name to load without extension
        
    Returns
    -------
    df: pandas DataFrame
        pandas data frame with the corresponing data set
        
    """
    
    os.chdir(path)
    fname = pollutant+'.pkl'
    print('Loading '+ pollutant +' file from: '+ path +'\n')
    df = pd.read_pickle(fname)
       
    return df
#%%

def create_dataSets(path):
    """
    Create and save data sets from csv files.
    The created data frames can be loaded by load_dataSets() function

    Parameters
    ----------
    path: str
        directory for loading

    Returns
    -------
    Pandas data sets

    """
    ### read csv files and save data frames
    
    Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N = Load_Pollutants_csv(path)
    ### saving created data frames
    save_df(path,Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N)

    return Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N



def load_dataSets(path):
    """
    Loads data set from csv files
    The files are inside the folder: "dataFrame_files"

    Parameters
    ----------
    path : str
        path to data set folder

    Returns
    -------
    None.

    """
    df = []
    pollutants = ['Ref_BC', 'Ref_O3', 'Ref_NO2', 'Ref_NO', 'Ref_PM10', 'Ref_N', 'Ref_Meteo', 
                  'LCS_O3', 'LCS_NO2', 'LCS_NO','LCS_Meteo',
                  'LCS_PM1','LCS_PM25','LCS_PM10','LCS_N']
    for p in pollutants:
        df.append(load_pkl(path, p))
    
    Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N  = [df[i] for i in range(0,len(df))] 
    
    return Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N
        
    
#%% aggregate measurements
def make_dataset(Ref_BC,Ref_O3,Ref_NO2,Ref_NO,Ref_N,Ref_PM,Ref_Meteo,LCS_O3,LCS_NO2,LCS_NO,LCS_N,LCS_PM1,LCS_PM25,LCS_PM10,LCS_Meteo,freq='10min',include_refMeasurements = True):
    """
    Aggregate measurements for every pollutant every X minutes, specified in freq

    Parameters
    ----------
    Ref_BC : TYPE
        DESCRIPTION.
    Ref_O3 : TYPE
        DESCRIPTION.
    Ref_NO2 : TYPE
        DESCRIPTION.
    Ref_NO : TYPE
        DESCRIPTION.
    Ref_N : TYPE
        DESCRIPTION.
    Ref_PM : TYPE
        DESCRIPTION.
    Ref_Meteo : TYPE
        DESCRIPTION.
    LCS_O3 : TYPE
        DESCRIPTION.
    LCS_NO2 : TYPE
        DESCRIPTION.
    LCS_NO : TYPE
        DESCRIPTION.
    LCS_N : TYPE
        DESCRIPTION.
    LCS_PM1 : TYPE
        DESCRIPTION.
    LCS_PM25 : TYPE
        DESCRIPTION.
    LCS_PM10 : TYPE
        DESCRIPTION.
    LCS_Meteo : TYPE
        DESCRIPTION.
    freq : str
        frequency to average measurements, default is 10min
    include_refMeasurements : bool
        Include reference station measurements in the data set. The default is True

    Returns
    -------
    ds : pandas DataFrame
        data set containing measurements every [freq] minutes for all the contaminants as columns.

    """
    df_BC = Ref_BC.groupby(pd.Grouper(key='date',freq=freq)).mean()
    df_O3 = Ref_O3.groupby(pd.Grouper(key='date',freq=freq)).mean()
    df_NO2 = Ref_NO2.groupby(pd.Grouper(key='date',freq=freq)).mean()
    df_N = Ref_N.groupby(pd.Grouper(key='date',freq=freq)).mean()
    df_Meteo = Ref_Meteo.groupby(pd.Grouper(key='date',freq=freq)).mean()
    df_NO = Ref_NO.groupby(pd.Grouper(key='date',freq=freq)).mean()
    df_PM = Ref_PM.groupby(pd.Grouper(key='date',freq=freq)).mean()
    
    #lcs
    df_O3_lcs = LCS_O3.groupby(pd.Grouper(key='date',freq=freq)).mean()
    df_NO2_lcs = LCS_NO2.groupby(pd.Grouper(key='date',freq=freq)).mean()
    df_NO_lcs = LCS_NO.groupby(pd.Grouper(key='date',freq=freq)).mean()
    df_N_lcs = LCS_N.groupby(pd.Grouper(key='date',freq=freq)).mean()
    df_N_lcs = df_N_lcs.iloc[:,0:3]#keep only ufp channel size that are also measured by the reference station
    #df_PM1_lcs = LCS_PM1.groupby(pd.Grouper(key='date',freq=freq)).mean()
    #df_PM25_lcs = LCS_PM25.groupby(pd.Grouper(key='date',freq=freq)).mean()
    df_PM10_lcs = LCS_PM10.groupby(pd.Grouper(key='date',freq=freq)).mean()
    df_Meteo_lcs = LCS_Meteo.groupby(pd.Grouper(key='date',freq=freq)).mean()
    
    
    print('Concatenating data set')
    if include_refMeasurements:
        ds = pd.concat([df_BC,df_O3,df_NO2,df_NO,df_N,df_PM,df_Meteo,df_O3_lcs,df_NO2_lcs,df_NO_lcs,df_N_lcs,df_PM10_lcs,df_Meteo_lcs],axis=1)
    else:
        ds = pd.concat([df_BC,df_Meteo,df_O3_lcs,df_NO2_lcs,df_NO_lcs,df_N_lcs,df_PM10_lcs,df_Meteo_lcs],axis=1)

    print('Removing NaN entries')
    ds.dropna(inplace=True)
    print('Cleaning wrong measurements')
    
    return ds

def sample_times(ds):
    print('Selecting measurement date')
    date_start = '2021-10-19 08:00:00'
    date_end = '2021-12-26 00:00:00'
    date_range = pd.date_range(start=date_start,end=date_end,freq='10T')
    ds = ds.loc[ds.index.isin(date_range)]
    
    return ds

    
    
    
    
#%% main
def main():
    """
    test data sets cration and loading
    create bool controls either the creation from csv files or loading dataFrames saved as .pkl files
    """
    
    path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/1_Files/raw_data_files'
    os.chdir(path)
    print('Changing directory to %s'%path)
    create = False
    
    if create:
        print('Testing .csv files loading and dataFrames creation')
        Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N = create_dataSets(path)
    else:
        Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N = load_dataSets(path)
        
        
    return Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N

if __name__=='__main__':
    Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N = main()

