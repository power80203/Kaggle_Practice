#!/usr/bin/env python3
"""
datareader has three main function as below

1. read data
2. check raw data property
3. return data to the FE part

"""


from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import Counter
np.seterr(divide='ignore', invalid='ignore')
sys.path.append(os.path.abspath(".."))

import config
import data.utilities as u_data

def dataReaderMain():


    pd.options.mode.chained_assignment = None

    train = pd.read_csv(config.trainset_path)
    test = pd.read_csv(config.testset_path)
    IDtest = test["PassengerId"]

    #########################################################
    #check and DROP!!! outliers

    Outliers_to_drop = u_data.detect_outliers(
        train, 2, ["Age", "SibSp", "Parch", "Fare"])

    print("######################################")
    print("take a look on outliers")
    print("######################################")
    print(train.loc[Outliers_to_drop])  # Show the outliers rows


    train = train.drop(Outliers_to_drop, axis=0).reset_index(drop=True)


    train_len = len(train)


    #########################################################
    #read data

    df = u_data.combineDataSet(train, test)

    # check dataframe datatype & describe
    # print(df.dtypes)
    # print(df.describe())

    #########################################################
    #check NA
    u_data.checkNA(df)



    return df, train, test, train_len, IDtest

if __name__ == "__main__":
    pass
