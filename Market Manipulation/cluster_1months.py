# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 11:07:10 2017

@author: Peng
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import pylab
matplotlib.style.use('ggplot')


n=0
path1 = 'E:\\Peng\\CUFE\\科研\\金融数据流挖掘\\新建文件夹\\数据\\training_data'
path2 = 'E:\\Peng\\CUFE\\科研\\金融数据流挖掘\\新建文件夹\\数据\\fig'

os.chdir(path1)
codes = ['603166', '300100', '002537']
mani_times = [['2016-07-05', '2016-07-18'],
              ['2014-11-26','2014-11-28'], 
              ['2014-09-26','2014-09-30'],]
training_time = [['2016-07-01', '2016-07-31'],
              ['2014-11-15','2014-12-15'], 
              ['2014-09-15','2014-10-15'],]
code = codes[n]
df = pd.read_csv(code+'.csv', index_col=['date'])

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

print("total_point {}".format(len(df)))


def label_set(y, rate, min_pro=0.01):    
    labels = list(set(y))
    for i in range(len(set(y))):
       if rate[i] <= min_pro:
           y[y==labels[i]] = 0
       else:
           y[y==labels[i]] = 1
    return y 

#KMeans
from sklearn.cluster import KMeans
y_kmeans = KMeans(n_clusters=2, random_state=10).fit_predict(X)
km_rate = [len(y_kmeans[y_kmeans==i]) / len(y_kmeans) for i in set(y_kmeans) ]

data_km = df.copy()
data_km['label'] = y_kmeans
out_km = data_km[data_km.label==1]['price']
out_rate_km = len(out_km[mani_times[n][0]:mani_times[n][1]]) / len(out_km)

print("K-means_out_number {}".format(len(out_km)))
print("k-means_accurate {}".format(out_rate_km))

#DBSCAN
from sklearn.cluster import DBSCAN
y_db = DBSCAN(eps=0.3, min_samples=10).fit(X).labels_
rate_db = [len(y_db[y_db==i]) / len(y_db) for i in set(y_db) ]

label_db = label_set(y_db, rate_db)
rate_label_db = [len(label_db[label_db==i]) / len(label_db) for i in set(label_db) ]

data_db = df.copy()
data_db['label'] = label_db
out_db = data_db[data_db.label==0]['price']
out_rate_db = len(out_db[mani_times[n][0]:mani_times[n][1]]) / len(out_db)

print("DBSCAN_out_number {}".format(len(out_db)))
print("DBSCAN_accurate {}".format(out_rate_db))
  

#Meanshift
from sklearn.cluster import MeanShift
y_ms = MeanShift().fit(X).labels_                
rate_ms = [len(y_ms[y_ms==i]) / len(y_ms) for i in set(y_ms) ]
label_ms = label_set(y_ms, rate_ms)
rate_label_ms = [len(label_ms[label_ms==i]) / len(label_ms) for i in set(label_ms) ]
rate_label_db = [len(label_db[label_db==i]) / len(label_db) for i in set(label_db) ]  
data_ms = df.copy()
data_ms['label'] = label_ms
out_ms = data_ms[data_ms.label==0]['price']
out_rate_ms = len(out_ms[mani_times[n][0]:mani_times[n][1]]) / len(out_ms)

print("MeanShift_out_number {}".format(len(out_ms)))
print("MeanShift_accurate {}".format(out_rate_ms))


#def plot_month(data,out,sta, end, name):
#    fig = plt.figure(figsize=(10,5))
#    data.loc[sta:end]['price'].plot()
#    plt.scatter(x=out.loc[sta:end].index, 
#                y=out.loc[sta:end],c='B')
#    plt.ylabel('Price')
#    path_to_save = os.path.join(path2,'({}){} to {}.png'.format(name,sta,end) )
#    fig.savefig(path_to_save)
#    plt.show()
#    
#plot_month(data_km, out_km,training_time[n][0],training_time[n][1],'K-means')
#plot_month(data_db, out_db,training_time[n][0],training_time[n][1],'DBSCAN')
#plot_month(data_ms, out_ms,training_time[n][0],training_time[n][1],'MeanShift')

sta = '2016-07-01'
end = '2016-07-31' 


##plot K-means
fig = plt.figure(figsize=(10,5))
data_km.loc[sta:end]['price'].plot(c='k')
plt.scatter(x=out_km.loc[sta:end].index, y=out_km.loc[sta:end],
            c='k', marker='^', label='anormal point')
plt.title('Detecting price manipution by K-means')
pylab.ylabel('price')
plt.legend(loc="lower right")
fig.savefig('(Kmeans) 07.01-07-31.png',dpi=fig.dpi)
plt.show()

##plot DBSCAN
fig = plt.figure(figsize=(10,5))
data_db.loc[sta:end]['price'].plot(c='k')
plt.scatter(x=out_db.loc[sta:end].index, y=out_db.loc[sta:end],
            c='k', marker='^', label='anormal point')
plt.title('Detecting price manipution by DBSCAN')
pylab.ylabel('price')
plt.legend(loc="lower right")
fig.savefig('(DBSCAN) 07.01-07-31.png',dpi=fig.dpi)
plt.show()

##plot Meanshift
fig = plt.figure(figsize=(10,5))
data_ms.loc[sta:end]['price'].plot(c='k')
plt.scatter(x=out_ms.loc[sta:end].index, y=out_ms.loc[sta:end],
            c='k', marker='^', label='anormal point')
plt.title('Detecting price manipution by MeanShift')
pylab.ylabel('price')
plt.legend(loc="lower right")
fig.savefig('(MeanShift) 07.01-07-31.png',dpi=fig.dpi)
plt.show()