#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import StandardScaler


sys.path.append(os.path.abspath(".."))

import config
import data.datareader as datareader
import data.utilities as u_data


def featureEngineeringMain():
    """
    1. 針對不同參數的不同的遺失值做處理
    1.a 畫圖確認要怎麼處理
    2. 製作ML所需的指標
    3. 吐出資料集到data/interi

    """

    # import data from datareader
    df2, train, test,train_len2, IDtest = datareader.dataReaderMain()

    df = df2

    train_len = train_len2

    #########################################################
    # dealing with missgin data
    #########################################################

    print("######################################")
    print("start dealing with missing data")
    print("######################################")
        
    #########################################################
    # Age

    """
    a important skill to fill out those NA value 
    is using other X vars

    """
    # age 是 nan的 index
    index_NaN_age = list(df["Age"][df["Age"].isnull()].index)

    # 把所有 Nan 作不同處理
    # 如果可以用Sibsp, parch, pclass 對到的話就用 對到的欄位去做中位數; 反之則用 全部的中位數去插捕
    for i in index_NaN_age: 
        age_med = df["Age"].median() #age_med中位數
        age_pred = df["Age"][((df['SibSp'] == df.iloc[i]["SibSp"]) & (
            df['Parch'] == df.iloc[i]["Parch"]) & (df['Pclass'] == df.iloc[i]["Pclass"]))].median()
        if not np.isnan(age_pred): 
            df['Age'].iloc[i] = age_pred
        else:
            df['Age'].iloc[i] = age_med

    scaler = StandardScaler()

    df['Age'] = scaler.fit_transform(df['Age'].values.reshape(-1,1))
   

    #Fare

    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    scaler = StandardScaler()

    df["Fare"] = scaler.fit_transform(df["Fare"].values.reshape(-1,1))

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

    # final check na status
    u_data.checkNA(df)

    #########################################################
    # Feature enginnering
    #########################################################

    print("######################################")
    print("start Feature enginnering process")
    print("######################################")

    #########################################################
    # Encoding categorical data

    # convert Sex into categorical value 0 for male and 1 for female
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})


    # Convert to categorical values Title

    df_title = [i.split(",")[1].split(".")[0].strip()
                for i in df["Name"]]
    df["Title"] = pd.Series(df_title)

    df["Title"] = df["Title"].replace(['Lady', 'the Countess', 'Countess', 'Capt',
                                    'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df["Title"] = df["Title"].map(
        {"Master": 0, "Miss": 1, "Ms": 1, "Mme": 1, "Mlle": 1, "Mrs": 1, "Mr": 2, "Rare": 3})
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


    


    #########################################################
    #pass completed dataframe to other.py#
    #########################################################

    print("######################################")
    print("Feature enginnering process was done")
    print("######################################")
    
    df.to_csv("test.csv")

    return df, train_len, IDtest
    
    


if __name__ == "__main__":
    #test functions
    featureEngineeringMain()


# ref:https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling
