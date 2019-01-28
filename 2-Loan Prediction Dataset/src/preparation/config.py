import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

pconfig = Path(__file__).parents[2]


#aa = os.path.abspath(os.path.join(yourpath, os.pardir))



traindatapath = "%s/data/raw/train.csv"%pconfig

testdatapath = "%s/data/raw/test.csv"%pconfig


if __name__ == "__main__":
    print("訓練資料集在", traindatapath)
    print(" ")
    print("測試資料集在", testdatapath)