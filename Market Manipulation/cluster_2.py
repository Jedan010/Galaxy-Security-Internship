# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 13:51:04 2017

@author: Peng
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import pylab
import matplotlib
matplotlib.style.use('ggplot')

paht1 = path1 = 'E:\\Peng\\CUFE\\科研\\金融数据流挖掘\\新建文件夹\\数据\\training_data'
os.chdir(path1)

df = pd.read_csv('603166_6-8.csv', index_col=['date'])
#df =pd.read_csv('603166.csv', index_col=['date'])
df.index = pd.to_datetime(df.index)
#df.sort_index(inplace=True)
df['type'] = df['type'].astype('category')
df['type'].cat.categories = [1, 0, -1]
df.loc[df.change=='--','change'] = 0
      


X = np.array(df)
#StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X[:,0:3] = scaler.fit_transform(X[:,0:3])



#KMeans
from sklearn.cluster import KMeans
y_kmeans = KMeans(n_clusters=2, random_state=10).fit_predict(X)
km_rate = [len(y_kmeans[y_kmeans==i]) / len(y_kmeans) for i in set(y_kmeans) ]

data_km = df.copy()
data_km['label'] = y_kmeans
out_km = data_km[data_km.label==1]['price']
out_rate_km = len(out_km['2016-07-05':'2016-07-18']) / len(out_km)
print("total_point {}".format(len(df)))
print("K-means_out_number {}".format(len(out_km)))
print("k-means_accurate {}".format(out_rate_km))

period = [['2016-06-01','2016-08-31'],
          ['2016-07-01','2016-07-31'],
          ['2016-07-05','2016-07-18'],
          ['2016-07-05','2016-07-12'],
          ['2016-07-12','2016-07-18'],
          ['2016-07-07 13:00:00','2016-07-07 15:00:00'],
          ['2016-07-07 13:18:00','2016-07-07 13:21:00'],
          ['2016-07-07 13:28:00','2016-07-07 13:30:00'],
          ['2016-07-07 13:30:00','2016-07-07 13:33:00'],
          ['2016-07-07 13:59:00','2016-07-07 14:01:00'],
          ['2016-07-11 13:00:00','2016-07-11 15:00:00'],
          ['2016-07-11 13:53:00','2016-07-11 14:03:00'],
          ]
#sta = '2016-07-01'
#end = '2016-07-31' 
def fig_save(data, out, sta, end, name):
    try:
        fig = plt.figure(figsize=(10,5))
        data.loc[sta:end]['price'].plot(c='k')
        plt.scatter(x=out.loc[sta:end].index, y=out.loc[sta:end],c='k',marker='^')
        pylab.ylabel('price')
        fig.savefig('({}){}——{}.png'.format(name, sta.replace(':','.'),
                      end.replace(':','.')),dpi=fig.dpi)
        plt.ylabel('Price')
        plt.show()
    except:
        print(sta, end)
        pass
    

    
def label_set(y, rate, min_pro=0.01):    
    labels = list(set(y))
    for i in range(len(set(y))):
       if rate[i] <= min_pro:
           y[y==labels[i]] = 0
       else:
           y[y==labels[i]] = 1
    return y 

##picture
#for day in period:
sta = period[0][0]
end = period[0][1]
fig_save(data_km, out_km, sta, end, 'KMeans')
    
#Meanshift
from sklearn.cluster import MeanShift
y_ms = MeanShift().fit(X).labels_                
rate_ms = [len(y_ms[y_ms==i]) / len(y_ms) for i in set(y_ms) ]    
            
label_ms = label_set(y_ms, rate_ms)
rate_label_db = [len(label_ms[label_ms==i]) / len(label_ms) for i in set(label_ms) ]                

data_ms = df.copy()
data_ms['label'] = label_ms
out_db = data_ms[data_ms.label==0]['price']
out_rate_db = len(out_db['2016-07-05':'2016-07-18']) / len(out_db)
print(len(out_db))
print(out_rate_db)
              
#for day in period:
#    fig_save(data_ms, out_db, day[0], day[1], 'MeanShift')








  
#DBSCAN
from sklearn.cluster import DBSCAN
y_db = DBSCAN(eps=0.3, min_samples=10).fit(X).labels_
rate_db = [len(y_db[y_db==i]) / len(y_db) for i in set(y_db) ]

label_db = label_set(y_db, rate_db)
rate_label_db = [len(label_db[label_db==i]) / len(label_db) for i in set(label_db) ]

data_db = df.copy()
data_db['label'] = label_db
out_db = data_db[data_db.label==0]['price']
out_rate_db = len(out_db['2016-07-05':'2016-07-18']) / len(out_db)
print("total_point {}".format(len(df)))
print("DBSCAN_out_number {}".format(len(out_db)))
print("DBSCAN_accurate {}".format(out_rate_db))

def plot_month(data,out, name):
    fig = plt.figure(figsize=(10,5))
    data.loc['2016-07-01':'2016-07-31']['price'].plot()
    plt.scatter(x=out.loc['2016-07-01':'2016-07-31'].index, 
                y=out.loc['2016-07-01':'2016-07-31'],c='B')
    plt.ylabel('Price')
    fig.savefig('{}.png'.format(name))
    plt.show()

plot_month(data_km, out_km, 'K-means')