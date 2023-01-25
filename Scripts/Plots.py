#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 15:07:50 2022

@author: jparedes
"""

import seaborn as sns
import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset

#%%
def plot_timeSeries(df_tr,df_val,df_ts,var='BC',save_fig=False,fname ='RefStation_60min.pdf'):
    mpl.rcParams.update({'text.usetex': False,'font.size': 45,
                         'mathtext.default':'regular','ytick.labelsize':'large',
                         'axes.labelsize':'large',
                         'axes.labelpad': 4.0})
    figx = 30
    figy = 25
    
    fig = plt.figure(figsize=(figx,figy))
    ax1 = fig.add_subplot(111)
    ax1.plot(df_tr.index,df_tr.loc[:,var],color = '#1a5276',label='Training set')
    ax1.plot(df_val.index,df_val.loc[:,var],color = '#1d8348',label='Validation set')
    ax1.plot(df_ts.index,df_ts.loc[:,var],color = '#d68910',label='Testing set')
    
    # y axis
    ymin = 0.
    ymax = 20.#max(pd.concat([df_tr.loc[:,var],df_val.loc[:,var],df_ts.loc[:,var]],axis=0))+1.
    yrange=np.arange(ymin,ymax,2.)
    ax1.set_yticks(ticks=yrange)
    
    ylabels = [str(np.round(i,2)) for i in yrange]
    ax1.set_yticklabels(labels=ylabels)
    ax1.set_ylabel(f'{var} ($\mu$g/m$^3$)',rotation=90)
  
    # x axis
    xrange = [df_tr.index[0],df_val.index[0],df_ts.index[0],df_ts.index[-1]]
    xrange = ['2018-01-01','2019-01-01','2019-07-01','2020-01-01'] # only for data set with dropped measurements
    ax1.set_xticks(ticks=xrange)
    ax1.set_xlim(left=17468.5,right=18298.5)# only for data with dropped measurements
    
    
    #xlabels = [str(int(i))+'%' for i in xrange]
    #ax1.set_xticklabels(labels=xlabels)
    ax1.set_xlabel(' ')
    
    # params
    ax1.legend(loc='upper right',prop={'weight':'normal'},ncol=1,
               framealpha=0.3,edgecolor = 'black',
               handleheight = 1,handletextpad=0.2)
    
    ax1.tick_params(direction='out', length=4, width=1)
    ax1.tick_params(axis='both', which='major')
    ax1.tick_params(axis='x',rotation=30,size=35)
    ax1.grid(alpha=0.5)
    ax1.set_title(f'{var} time series')
    
    
    for axis in ['top','bottom','left','right']:
      ax1.spines[axis].set_linewidth(2)
  
    #fig.tight_layout(pad=3.0)
    fig.set_size_inches(figx,figy)

    if save_fig:
        plt.savefig(fname,dpi=1200,format='pdf')

    # TO DO:
    #       generalize to any input
    return fig

def plot_original_predicted(df,save_fig=False,fname='BC_testingSet_original_vs_predictions.pdf'):
    """
    Plot Reference station real measurements vs proxy predictions
    """

    mpl.rcParams.update({'text.usetex': False,'font.size': 45,
                         'mathtext.default':'regular','ytick.labelsize':'large',
                         'axes.labelsize':'large',
                         'axes.labelpad': 4.0})
    figx = 30
    figy = 25
    
    fig = plt.figure(figsize=(figx,figy))
    ax1 = fig.add_subplot(111)
    ax1.plot(df.index,df.iloc[:,0],'o-',color = '#1a5276',label='Reference Station measurements')
    ax1.plot(df.index,df.iloc[:,1],'^-',color = 'orange',label='Predicted measurements')
    
    # y axis
    ymin = 0.
    ymax = df.max(axis=0).max()
    yrange=np.arange(ymin,ymax,2.)
    ax1.set_yticks(ticks=yrange)
    
    ylabels = [str(np.round(i,2)) for i in yrange]
    ax1.set_yticklabels(labels=ylabels)
    ax1.set_ylabel('BC ($\mu$g/m$^3$)',rotation=90)
  
    # x axis
    xrange = pd.date_range(start=df.index[0],end=df.index[-1],periods=4)
    ax1.set_xticks(ticks=xrange)
    
    #xlabels = [str(int(i))+'%' for i in xrange]
    #ax1.set_xticklabels(labels=xlabels)
    ax1.set_xlabel(' ')
    
    # params
    ax1.legend(loc='upper right',prop={'weight':'normal'},ncol=1,
               framealpha=0.3,edgecolor = 'black',
               handleheight = 1,handletextpad=0.2)
    
    ax1.tick_params(direction='out', length=4, width=1)
    ax1.tick_params(axis='both', which='major')
    ax1.tick_params(axis='x',rotation=30,size=35)
    ax1.grid(alpha=0.5)
    ax1.set_title('BC testing set time series')
    
    
    for axis in ['top','bottom','left','right']:
      ax1.spines[axis].set_linewidth(2)
  
    #fig.tight_layout(pad=3.0)
    fig.set_size_inches(figx,figy)

    if save_fig:
        plt.savefig(fname,dpi=1200,format='pdf')

    # TO DO:
    #       generalize to any input
    return fig
#%%
def main():
    print('Plotting script\n---------------')    
    return
#%%
if __name__ == '__main__':
    main()
