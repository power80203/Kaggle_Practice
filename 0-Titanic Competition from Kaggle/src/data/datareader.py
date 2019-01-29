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

train = pd.read_csv(config.trainset_path)
test = pd.read_csv(config.testset_path)


def combineDataSet(train, test):
    """
    pass

    """
    
    
    dataset = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

    return dataset


df = combineDataSet(train, test)

# check dataframe datatype & describe
print(df.dtypes)
print(df.describe())

###############################
# check data NA

# Fill empty and NaNs values with NaN
df = df.fillna(np.nan)

# Check for Null values
print(df.isnull().sum())


###############################
#Filling missing value

#Age

index_NaN_age = list(df["Age"][df["Age"].isnull()].index)

for i in index_NaN_age:
    age_med = df["Age"].median()
    age_pred = df["Age"][((df['SibSp'] == df.iloc[i]["SibSp"]) & (
        df['Parch'] == df.iloc[i]["Parch"]) & (df['Pclass'] == df.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred):
        df['Age'].iloc[i] = age_pred
    else:
        df['Age'].iloc[i] = age_med


# Cabin 

df["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in df['Cabin']])





# Create a family size descriptor from SibSp and Parch
df["Fsize"] = df["SibSp"] + df["Parch"] + 1



#dummie
df = pd.get_dummies(df, columns=["Ticket"], prefix="T")
    # Create categorical values for Pclass
df["Pclass"] = df["Pclass"].astype("category")
df = pd.get_dummies(df, columns=["Pclass"], prefix="Pc")
    # Drop useless variables
df.drop(labels=["PassengerId"], axis=1, inplace=True)


# ref:https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling


sys.exit()




X = dataset.drop(['PassengerId', 'Cabin', 'Ticket',
                  'Fare', 'Parch', 'SibSp'], axis=1)
y = X.Survived                       # vector of labels (dependent variable)
# remove the dependent variable from the dataframe X
X = X.drop(['Survived'], axis=1)


# ----------------- Encoding categorical data -------------------------

# encode "Sex"
labelEncoder_X = LabelEncoder()
X.Sex = labelEncoder_X.fit_transform(X.Sex)


# encode "Embarked"

# number of null values in embarked:
print('Number of null values in Embarked:', sum(X.Embarked.isnull()))

# fill the two values with one of the options (S, C or Q)
row_index = X.Embarked.isnull()
X.loc[row_index, 'Embarked'] = 'S'

Embarked = pd.get_dummies(X.Embarked, prefix='Embarked')
X = X.drop(['Embarked'], axis=1)
X = pd.concat([X, Embarked], axis=1)
# we should drop one of the columns
X = X.drop(['Embarked_S'], axis=1)





"""

if __name__ == "__main__":
    
    ###############################
    #drop NA

    df = df.fillna(value=0)

    print(df.head())

    print("dataframe ok !!")

###############################

# computing corr

def computingCorr(dataframe1, list1): 
    df_corr = dataframe1.loc[:, list1]
    return df_corr.corr()
    

# print(computingCorr(df, ["Age", "Survived"]))



# check normal status
# pass


###############################
# split sample

X_train = df.loc[:, ["Age", "Pclass"]].values.reshape(-1, 2)
X_test = df2.fillna(value=0).loc[:, ["Age", "Pclass"]].values.reshape(-1, 2)

y_train = df.loc[:, "Survived"].values.reshape(-1, 1)

print("X 訓練資料的維度為  ", X_train.shape)
print("X 測試資料的維度為  ", X_test.shape)

print("Y 訓練資料的維度為  ", y_train.shape)

###############################
#LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

y_predicted = model.predict(X_test)

# y_predicted = (y_predicted > 0.5)


def judgei(i):
    if i > 0.5:
        return 1
    else:
        return 0


df2['y_hats'] = y_predicted


df2['y_hats'] = df2['y_hats'].apply(lambda x: judgei(x))


df3 = df2.loc[:, ["PassengerId", "y_hats"]]

df3.columns = ["PassengerId", "Survived"]



df3.to_csv("test.csv", index = False)

print()
print('done')







"""








