# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 17:56:49 2017

@author: J
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from itertools import cycle
from scipy import interp



path1 = 'E:\\Peng\\CUFE\\科研\\金融数据流挖掘\\新建文件夹\\数据\\training_data'
os.chdir(path1)

data = pd.read_csv('1-5tick.csv',index_col=[1])
data.drop('Unnamed: 0', axis=1, inplace=True)


print("total_data: {}".format(len(data)))
print("out_data: {}".format(len(data[data.label==1])))
print("out_rate: {}".format(len(data[data.label==1]) / len(data)))

s = len(data)
n = s/100
k = n/100
num = [np.random.randint(i*(n/k), (i+1)*(n/k)) for i in range(int(s/100))]
df = data.iloc[num]

code_list = set(df['code'])

X = np.array(df.drop(['label','code'], 1))
y = np.array(df['label'])

from sklearn import preprocessing
X = preprocessing.scale(X)

print("simplified_data: {}".format(len(df)))
print("simplified_out_data: {}".format(len(df[df.label==1])))
print("simplified_out_rate: {}".format(len(df[df.label==1]) / len(df)))




#K-neighbours
from sklearn import neighbors
knh = neighbors.KNeighborsClassifier()

#DecisionTreeClassifier 
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()

#Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()

#Quadratic Discriminant Analysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
qda = QuadraticDiscriminantAnalysis()

#Neural network models (supervised)
from sklearn.neural_network import MLPClassifier
nn = MLPClassifier(solver='lbfgs', alpha=1e-5,
                  hidden_layer_sizes=(5, 2), random_state=1)


#logistic regression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression()

##Support Vector Classification
from sklearn import svm
svm_clf = svm.SVC(probability=True)



from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
cv = StratifiedKFold(n_splits=5)

def index(model):
    predicted = cross_val_predict(model, X, y, cv=cv)
    
    Confusion_Matrix = confusion_matrix(y, predicted)
    accuracy = metrics.accuracy_score(y, predicted) 
    sensitivity = Confusion_Matrix[0][0] / (Confusion_Matrix[0][0] + Confusion_Matrix[0][1])
    specificity = Confusion_Matrix[1][1] / (Confusion_Matrix[1][1] + Confusion_Matrix[1][0])
     
    print("Confusion_Matrix:")
    print(Confusion_Matrix)
    print("Accuracy :%f" %(accuracy))
    print("Sensitivity :%f" %(sensitivity))
    print("Specificity :%f" %(specificity))


models = [knh,dtc,lda,qda,nn,logr,svm_clf]
for model in models:
    index(model)
 

   



colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange','red'])
line_types = ['-', '--', '-.', ':', ',','+','*' ]
names = ['KNH', 'DTC', 'LDA', 'QDA', 'ANN', 'LR', 'SVM']
models = [knh,dtc,lda,qda,nn,logr,svm_clf]

from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

lw = 2

fig = plt.figure(figsize=(8,5))
plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Luck')

for model, line_type, name, color in zip(models, line_types, names, colors): 
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    for train, test in tscv.split(X, y):
        probas_ = model.fit(X[train], y[train]).predict_proba(X[test])
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
                
    mean_tpr /= tscv.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, line_type, color=color, 
             label='{}: {:.2f}'.format(name,mean_auc), lw=lw)
    print('{}:{}'.format(name,mean_auc ))
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
fig.savefig('ROC_tick_colorful.png', dpi=fig.dpi)
plt.show()
