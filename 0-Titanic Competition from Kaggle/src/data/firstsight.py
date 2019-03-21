import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(".."))

print(sys.path)
import config
import data.utilities as u
from visualization import scatterchart as sc
from visualization import barchart as bar



#########################################################
#Loading Data#
#########################################################
train = pd.read_csv("%s" % config.trainset_path)
test = pd.read_csv("%s" % config.testset_path)

print(train.shape)
print(test.shape)
print('')

train = train.drop('Survived',1)

print('removed target value')
print(train.shape)
print(test.shape)
print('')

# combine togther

df = train.append(test)

# train.drop(Outliers_to_drop, axis=0).reset_index(drop=True)

print("after combinition", df.shape)
print('')

u.checkNA(df)

#########################################################
#EDA check#
#########################################################

#########################################################
#check correlation

#check corrplot
# sc.scatterpure(df)


def sex_existed(df):

    figure = plt.figure(figsize=(15, 8))

    df = df.dropna()

    plt.hist([df[df["Sex"] =="male"]["Age"], df[df["Sex"] =="female"]["Age"]],\
      label=["Male", "Female"])
    plt.xlabel('Age')
    plt.ylabel('Number of Sex')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # sex_existed(df)
    bar.barchart(df, "Age", "Sex")
