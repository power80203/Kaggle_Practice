import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import Counter



def combineDataSet(train, test):
    """
    combine data

    """

    dataset = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

    return dataset

def checkNA(df):

    # Fill empty and NaNs values with NaN
    df = df.fillna(np.nan)
    # Check for Null values
    print(df.isnull().sum())


def detect_outliers(df, n, features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method (Tukey JW., 1977) .

    param:

    df = dataframe
    n = how many outliers
    features = feature we need to check

    """
    outlier_indices = []

    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step)
                              | (df[col] > Q3 + outlier_step)].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    return multiple_outliers

