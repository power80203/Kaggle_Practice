#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import time
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

sns.set(style='white', context='notebook', palette='deep')

sys.path.append(os.path.abspath(".."))

import config
import features.featureEngineering as FE

test_model = False

def trainModelMain():
    dataset, train_len, IDtest = FE.featureEngineeringMain()
    train = dataset[:train_len] # 以前 是 train
    test = dataset[train_len:] # 以後是 test
    test.drop(labels=["Survived"],axis = 1,inplace=True)

    #########################################################
    # Separate train features and label 

    train["Survived"] = train["Survived"].astype(int)

    Y_train = train["Survived"]

    X_train = train.drop(labels = ["Survived"],axis = 1)

    #########################################################
    # models


    kfold = StratifiedKFold(n_splits=10)

    random_state = 2

    classifiers = []
    #SVC
    classifiers.append(SVC(random_state=random_state))
    #DTree
    classifiers.append(DecisionTreeClassifier(random_state=random_state))
    #Ada
    classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
    #RF
    classifiers.append(RandomForestClassifier(random_state=random_state))
    #ETree
    classifiers.append(ExtraTreesClassifier(random_state=random_state))
    #GDBT
    classifiers.append(GradientBoostingClassifier(random_state=random_state))
    #MLP
    classifiers.append(MLPClassifier(random_state=random_state))
    #KNN
    classifiers.append(KNeighborsClassifier())
    #Sigmoild
    classifiers.append(LogisticRegression(random_state = random_state))
    #Linera Disc
    classifiers.append(LinearDiscriminantAnalysis())

    if test_model :
        cv_results = []
        for classifier in classifiers :
            cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

        cv_means = []
        cv_std = []
        for cv_result in cv_results:
            cv_means.append(cv_result.mean())
            cv_std.append(cv_result.std())

        cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
        "RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})


        #########################################################
        # plot resutl

        g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
        g.set_xlabel("Mean Accuracy")
        g = g.set_title("Cross validation scores")
        plt.show()

    #########################################################
    #start tunning models#
    #########################################################


    #########################################################
    #ADABOOST

    print("ADABOOST Process")

    DTC = DecisionTreeClassifier()
    adaDTC = AdaBoostClassifier(DTC, random_state=7)

    ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

    gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

    gsadaDTC.fit(X_train,Y_train)

    ada_best = gsadaDTC.best_estimator_

    #########################################################
    #  ExtraTrees
    ExtC = ExtraTreesClassifier()
    print()
    print("ExtraTrees Process")

    ## Search grid for optimal parameters
    ex_param_grid = {"max_depth": [None],
                "max_features": [1, 3, 10],
                "min_samples_split": [2, 3, 10],
                "min_samples_leaf": [1, 3, 10],
                "bootstrap": [False],
                "n_estimators" :[100,300],
                "criterion": ["gini"]}


    gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

    gsExtC.fit(X_train,Y_train)

    ExtC_best = gsExtC.best_estimator_

    #########################################################
    #  RandomForest
    RFC = RandomForestClassifier()
    print()

    print("RandomForest Process")

    ## Search grid for optimal parameters
    rf_param_grid = {"max_depth": [None],
                "max_features": [1, 3, 10],
                "min_samples_split": [2, 3, 10],
                "min_samples_leaf": [1, 3, 10],
                "bootstrap": [False],
                "n_estimators" :[100,300],
                "criterion": ["gini"]}


    gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

    gsRFC.fit(X_train,Y_train)

    RFC_best = gsRFC.best_estimator_

    #########################################################
    #GradientBoosting
    print()

    print("GradientBoosting Process")

    GBC = GradientBoostingClassifier()

    gb_param_grid = {'loss' : ["deviance"],
                'n_estimators' : [100,200,300],
                'learning_rate': [0.1, 0.05, 0.01],
                'max_depth': [4, 8],
                'min_samples_leaf': [100,150],
                'max_features': [0.3, 0.1] 
                }

    gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

    gsGBC.fit(X_train,Y_train)

    GBC_best = gsGBC.best_estimator_

    #########################################################
    # SVC classifier
    print()
    print("SVC Process")

    SVMC = SVC(probability=True)
    svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

    gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

    gsSVMC.fit(X_train,Y_train)

    SVMC_best = gsSVMC.best_estimator_


    #########################################################
    #Ensemble all models#
    #########################################################

    votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
    ('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=4)

    votingC = votingC.fit(X_train, Y_train)

    import pickle #pickle模块

    # store modle
    with open('%s/clf.pickle'%config.model_saving_path, 'wb') as f:
        pickle.dump(votingC, f)
    #读取Model
    with open('%s/clf.pickle'%config.model_saving_path, 'rb') as f:
        votingC = pickle.load(f)

    test_Survived = pd.Series(votingC.predict(test), name="Survived")

    results = pd.concat([IDtest,test_Survived],axis=1)

    results.to_csv("{}ensemble_python_voting_{}.csv".format(config.pred_result_path, time.time()),index=False)











    







if __name__ == "__main__":
    trainModelMain()