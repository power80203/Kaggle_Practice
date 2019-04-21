#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import models
from keras import layers
from keras import optimizers
# load models
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from keras.wrappers.scikit_learn import KerasClassifier


sns.set(style='white', context='notebook', palette='deep')

sys.path.append(os.path.abspath(".."))

import config
import features.featureEngineering as FE

test_model = False

def trainModelDNNMain():
    dataset, train_len, IDtest = FE.featureEngineeringMain()
    train = dataset[:train_len] # 以前 是 train
    test = dataset[train_len:] # 以後是 test
    test.drop(labels=["Survived"],axis = 1,inplace=True)

    #########################################################
    # Separate train features and label 

    train["Survived"] = train["Survived"].astype(int)

    Y_train = train["Survived"]

    X_train = train.drop(labels = ["Survived"],axis = 1)

    model = KerasClassifier(build_fn = create_model, verbose = 0 )

    #########################################################
    # set param_grid ,epochs, batch_size, activation
    param_grid = dict(epochs=[30,50,100], batch_size = [8,16],
    activation = ['relu', 'tanh', 'softmax', 'linear', 'hard_sigmoid', 'softplus', 'selu'])
    
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)

    grid_result = grid.fit(X_train, Y_train)

    print('Best: {} using {}'.format(grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, std, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, std, param))

    
def create_model(activation='relu'):

    model = models.Sequential()
    model.add(layers.Dense(12, activation = activation, input_shape=(60,)))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    
    return model

def trainModelDNNMain_final():

    dataset, train_len, IDtest = FE.featureEngineeringMain()
    train = dataset[:train_len] # 以前 是 train
    test = dataset[train_len:] # 以後是 test
    test.drop(labels=["Survived"],axis = 1,inplace=True)

    #########################################################
    # Separate train features and label 

    train["Survived"] = train["Survived"].astype(int)

    Y_train = train["Survived"]
    X_train = train.drop(labels = ["Survived"],axis = 1)
    
    model = models.Sequential()
    model.add(layers.Dense(60, activation = 'relu', input_shape=(60,)))
    model.add(layers.Dense(120, activation = 'relu'))
    model.add(layers.Dense(120, activation = 'relu'))
    model.add(layers.Dense(60, activation = 'relu'))
    model.add(layers.Dense(12, activation = 'relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    model.fit(X_train, Y_train, batch_size = 8, epochs = 100)
    
    #########################################################
    # export resut to csv
    y_pred = model.predict_classes(test)

    y_pred = np.reshape(y_pred,(len(y_pred),))
 
    results = pd.DataFrame({'PassengerId':IDtest, 'Survived':y_pred})

    #test_Survived = pd.Series(y_pred, name="Survived")
    # results = pd.concat([IDtest,test_Survived],axis=1)

    results.to_csv("Dnn_results.csv",index=False)    

if __name__ == "__main__":
#     trainModelDNNMain()
    trainModelDNNMain_final()
    pass