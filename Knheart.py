#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 23:56:49 2019

@author: Nikolas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import pickle


import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from tkinter import*
from sklearn.externals import joblib


dataset = pd.read_csv('heart.csv')
dataset3 = pd.read_csv('heart.csv')
dataset2=pd.read_csv('copia.csv')



dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

def copy_csv():
    dataset = pd.read_csv('heart.csv')
 #   df = dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
    dataset = dataset.head(0)
    dataset.to_csv('copia.csv'  ,index=False)
    
    
    

standardScaler = StandardScaler(with_mean=True, with_std=True)
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])


new_row={"age":12,"sex":1,"cp":1,"trestbps":1,
                    "chol":1,
                    "fbs":1,
                    "restecg":1,
                    "thalach":1,
                    "exang":1,
                    "oldpeak":1,
                    "slope":1,
                    "ca":1,
                    "thal":1,
                    "target":1
                    }
dataset2= dataset2.append(new_row, ignore_index=True)
dataset2 = pd.get_dummies(dataset2, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
dataset2=dataset2.reindex(columns = dataset.columns.values,fill_value=0)
#print(dataset['target'])
y = dataset['target']

X = dataset.drop(['target'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
knn_classifier = KNeighborsClassifier(n_neighbors = 8)
knn_classifier.fit(X_train, y_train)
pickle.dump(knn_classifier, open('model.pkl','wb'))# Loading model to compare the results


knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X_train, y_train)
    knn_scores.append(knn_classifier.score(X_test, y_test))

plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')
for i in range(1,21):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1, 21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')
print("The score for K Neighbors Classifier is {}% with {} nieghbors.".format(knn_scores[7]*100, 8))

