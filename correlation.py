import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def corr_graph(Data,title):
    corr=Data.corr()
    corr=corr.pow(2)
    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr,cmap='coolwarm', vmin=0, vmax=1)
    ticks = np.arange(0,len(Data.columns),1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(Data.columns,fontsize=40,rotation=-50)
    ax.set_yticklabels(Data.columns,fontsize=40)
    ax.xaxis.set_ticks_position('bottom')
    cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.77])
    cb = fig.colorbar(cax,cax=cbar_ax)
    cb.ax.tick_params(labelsize=40)
    ax.set_title(title,fontsize=40, fontweight='bold')
    corr=Data.corr()
    corr=corr.pow(2)
    arr = corr.values
    index_names = corr.index
    col_names = corr.columns
    R,C = np.where(np.triu(arr,1)>0.9)
    out_arr = np.column_stack((index_names[R],col_names[C],arr[R,C]))
    df_corr = pd.DataFrame(out_arr,columns=[['row_name','col_name','R2']])
    plt.savefig('./correlation analysis/ANSA/ANSA_AD.png', dpi = 300)
    return(df_corr)

titles = ['BHA','SMC','ANSA']
file = './correlation analysis/ANSA/AD.csv'
Data = pd.read_csv(file)
title = 'ANSA'
corr_graph(Data,title)

