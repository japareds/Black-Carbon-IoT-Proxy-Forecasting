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

#%%
def plot_timeSeries(df,var='BC'):
    mpl.rcParams.update({'text.usetex': False,'font.size': 20,
                         'mathtext.default':'regular','ytick.labelsize':'large',
                         'axes.labelsize':'large',
                         'axes.labelpad': 4.0})
    figx = 30
    figy = 18
    

    fig = plt.figure(figsize=(figx,figy))
    ax1 = sns.lineplot(x=df.index,y=df.loc[:,var])
    
    # y axis
    ymin = 0.
    ymax = max(df.loc[:,var])
    yrange=np.arange(ymin,ymax,2.)
    ax1.set_yticks(ticks=yrange)
    
    ylabels = [str(np.round(i,2)) for i in yrange]
    ax1.set_yticklabels(labels=ylabels)
    ax1.set_ylabel('BC ($\mu$g/m$^3$)',rotation=90)
  
    # x axis
    xrange = pd.date_range(start=df.index[0],end=df.index[-1],freq='6MS')
    ax1.set_xticks(ticks=xrange)
    
    #xlabels = [str(int(i))+'%' for i in xrange]
    #ax1.set_xticklabels(labels=xlabels)
    ax1.set_xlabel(' ')
    
    # params
    ax1.legend(loc='upper left',prop={'weight':'normal'},ncol=1,
               framealpha=0.3,edgecolor = 'black',
               handleheight = 1,handletextpad=0.2)
    
    ax1.tick_params(direction='out', length=4, width=1)
    ax1.tick_params(axis='both', which='major')
    ax1.grid(alpha=0.5)
    ax1.set_title(f'{var} time series')
    
    
    for axis in ['top','bottom','left','right']:
      ax1.spines[axis].set_linewidth(2)
  
    #fig.tight_layout(pad=3.0)
    fig.set_size_inches(figx,figy)

    return fig
#%%
def main():
    
    return
#%%
if __name__ == '__main__':
    main()
