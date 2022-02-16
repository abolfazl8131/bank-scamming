# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 02:50:53 2022

@author: abolfazl81
"""

from sklearn.preprocessing import MinMaxScaler,Normalizer,StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV,ShuffleSplit
from pandas import set_option
from numpy import set_printoptions
from sklearn.metrics import mean_squared_error,accuracy_score, r2_score
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression,Ridge,RidgeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import BaggingRegressor,RandomForestRegressor,GradientBoostingRegressor,BaggingClassifier
import urllib
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.linear_model import SGDClassifier,PassiveAggressiveClassifier
from sklearn.neighbors import NearestCentroid
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,hinge_loss,log_loss,confusion_matrix
from mlxtend.evaluate import bias_variance_decomp
#load a dataset we use for training
df = pd.read_csv('bank-additional-full.csv', sep=';')
X = df.iloc[:,:20]
y = df.iloc[:,20]

#encode lables using one hot encode
X = pd.get_dummies(X)
#scaling
scaler = StandardScaler()

X = scaler.fit_transform(X)


#using the model for classification
model = DecisionTreeClassifier()

#encode y lables --- 1=>Yes && 0=>No
label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

#dividing to train-test set but we use it for training // 
#a slice of data is enaugh for training
#Because second data set we use for real testing is subset of this dataset
#we should avoid noise in the model
X_train,X_test,y_train,y_test = train_test_split(X,
                                                 y,
                                                 train_size=0.8,
                                                 test_size=0.2,
                                                 shuffle=True)



#loading test data
df_test = pd.read_csv('bank-additional.csv', sep=';')
#using another dataset for predict,,,, it is not from training set!
X_real = df_test.iloc[:,:20]
#encode...
X_real = pd.get_dummies(X_real)

y_real =  df_test.iloc[:,20]

y_real = label_encoder.transform(y_real)
# using previous training slices to train the model
model.fit(X_train,y_train)
#transform a real X set
X_real = scaler.transform(X_real)
#predicting
prediction = model.predict(X_real)
#get the accuracy of our model via comparing real-y and predicted-y
accuracy_of_real_data = accuracy_score(y_real, prediction)
#making confusion matrix
matrix = confusion_matrix(y_real, prediction)
# this one is used for calculationg bias and variance
#bias = error in training slice
#test/validation-error = error in test/validation slice
#variance = test/validation-error - bias
avg_loss,avg_bias,avg_var = bias_variance_decomp(model, 
                                                 X_train,
                                                 y_train,
                                                 X_real,
                                                 y_real,
                                                 loss='0-1_loss',
                                                 num_rounds=20)

print('bias:',avg_bias)

print('variance:',avg_var )

print('accuracy_in_real:',accuracy_of_real_data )

print('loss:',avg_loss)




