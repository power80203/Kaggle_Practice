import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(".."))

import config
import data.datareader as datareader
import data.utilities as u_data
import data.utilities as u
from visualization import scatterchart as sc
from visualization import barchart as bar



#########################################################
#Loading Data#
#########################################################

df2, train, test,train_len2, IDtest = datareader.dataReaderMain()

df = df2

train_len = train_len2


# print(df.head())

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
    # bar.barchart(df, "Age", "Sex")

    print(df.columns)
