#!/usr/bin/env python3
"""
datareader has three main function as below

1. read data
2. check raw data property
3. TBD

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

pd.options.mode.chained_assignment = None

train = pd.read_csv(config.trainset_path)
test = pd.read_csv(config.testset_path)
IDtest = test["PassengerId"]

#########################################################
#check outliers

Outliers_to_drop = u.detect_outliers(train, 2, ["Age", "SibSp", "Parch", "Fare"])
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
u.checkNA(df)

#########################################################
# filling NA

"""
a important skill to fill out those NA value 
is using other X vars 

"""

#Age

# age 是 nan的 index
index_NaN_age = list(df["Age"][df["Age"].isnull()].index)


# 把所有 Nan 作不同處理
# 如果可以用Sibsp, parch, pclass 對到的話就用 對到的欄位去做中位數; 反之則用 全部的中位數去插捕
for i in index_NaN_age:
    age_med = df["Age"].median()
    age_pred = df["Age"][((df['SibSp'] == df.iloc[i]["SibSp"]) & (
        df['Parch'] == df.iloc[i]["Parch"]) & (df['Pclass'] == df.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred):
        df['Age'].iloc[i] = age_pred
    else:
        df['Age'].iloc[i] = age_med


#Fare

df["Fare"] = df["Fare"].fillna(df["Fare"].median())


# Cabin 
df["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in df['Cabin']])

# Ticket
Ticket = []
for i in list(df.Ticket):
    if not i.isdigit():
        Ticket.append(i.replace(".", "").replace(
            "/", "").strip().split(' ')[0])  # Take prefix
    else:
        Ticket.append("X")

df["Ticket"] = Ticket


# "Embarked"
df["Embarked"] = df["Embarked"].fillna("S")

# ----------------- Encoding categorical data -------------------------

# convert Sex into categorical value 0 for male and 1 for female
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Convert to categorical values Title 

df_title = [i.split(",")[1].split(".")[0].strip()
                 for i in df["Name"]]
df["Title"] = pd.Series(df_title)

df["Title"] = df["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df["Title"] = df["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
df["Title"] = df["Title"].astype(int)

# Create a family size descriptor from SibSp and Parch
df["Fsize"] = df["SibSp"] + df["Parch"] + 1



# convert to indicator values Title and Embarked
df = pd.get_dummies(df, columns=["Title"])
df = pd.get_dummies(df, columns=["Embarked"], prefix="Em")
df = pd.get_dummies(df, columns=["Ticket"], prefix="T")
df = pd.get_dummies(df, columns=["Cabin"])

#########################################################
#drop useless


# Drop useless variables
df.drop(["PassengerId"], axis=1, inplace=True)
df.drop(labels=["Name"], axis=1, inplace=True)


print(df.head())




#########################################################
#pass completed dataframe to main.py#
#########################################################

def getData():
    return df, train_len
    


if __name__ == "__main__":
    pass


# ref:https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling
