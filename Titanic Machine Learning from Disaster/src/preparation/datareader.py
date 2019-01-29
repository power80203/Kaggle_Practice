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
import preparation.utilities as utilities


train = pd.read_csv(config.trainset_path)
test = pd.read_csv(config.testset_path)
IDtest = test["PassengerId"]


Outliers_to_drop = utilities.detect_outliers(train, 2, ["Age", "SibSp", "Parch", "Fare"])

train = train.drop(Outliers_to_drop, axis=0).reset_index(drop=True)





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








