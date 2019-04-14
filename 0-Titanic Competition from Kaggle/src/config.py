#!/usr/bin/env python3


#########################################################
#讀取檔案
#########################################################

#csv file

import csv
import os
import sys

a = os.path.abspath(__file__)
a = a.split("/")
str_line = '/'
a = str_line.join(a[0:-2])



testset_path = r"{}/data/raw/test.csv".format(a)
trainset_path = r"{}/data/raw/train.csv".format(a)

#########################################################
#output info#
#########################################################

# hd5 model

model_saving_path = r"{}/models".format(a)

# prediction result

pred_result_path = r"{}/data/processed/".format(a)



if __name__ == "__main__":
    print(testset_path)
    print(model_saving_path)

