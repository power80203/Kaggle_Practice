#!/usr/bin/env python3

#########################################################
#讀取檔案
#########################################################

#csv file

from pathlib import Path
import csv
import os
import sys
sys.path.append(os.path.abspath("."))

p = Path(__file__).parents[1]


testset_path = r"{}/data/raw/test.csv".format(p)
trainset_path = r"{}/data/raw/train.csv".format(p)



"""
filelocation = "D:\\1-Project\\2018\\1-HLC\\3-Data\\DATA\\0828\\Student_Info_Dist.csv"
ecod= 'big5'
"""
#########################################################
#Random Forest parameter
#########################################################

"""
RF_testsize = 0.2

n_estimators = 200 # 幾棵樹

min_samples_split = 40

max_depth = 15

min_samples_leaf = 10

"""
