import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

sys.path.append(os.path.abspath(".."))

import data.datareader
import features.featureEngineering as FE


df, train_len, IDtest = FE.featureEngineeringMain()

# df, train, test,train_len, IDtest = data.datareader.dataReaderMain()

train = df[:train_len]
test = df[train_len:]
test.drop(labels=["Survived"], axis=1, inplace=True)


train["Survived"] = train["Survived"].astype(int)

Y_train = train["Survived"]

X_train = train.drop(labels=["Survived"], axis=1)

print(df.columns)
print(df.head())


def gradient_boosting():
    # Gradient boosting tunning

    GBC = GradientBoostingClassifier()
    gb_param_grid = {'loss': ["deviance"],
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.1, 0.05, 0.01],
                    'max_depth': [4, 8],
                    'min_samples_leaf': [100, 150],
                    'max_features': [0.3, 0.1]
                    }

    gsGBC = GridSearchCV(GBC, param_grid=gb_param_grid,
                        scoring="accuracy", n_jobs=4, verbose=0)


    grid_result = gsGBC.fit(X_train, Y_train)

    GBC_best = gsGBC.best_estimator_

    print('Best: {} using {}'.format(grid_result.best_score_, grid_result.best_params_))



    # Best score
    test_Survived = pd.Series(gsGBC.predict(test), name="Survived")

    print(test_Survived)




if __name__ == "__main__":
    pass