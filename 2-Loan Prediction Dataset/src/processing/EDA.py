############################################## 導入所需模組
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

p = Path(__file__).parents[2]
sys.path.append("%s"%p)

import src.preparation.dataReader as dataReader


############################################## 獲取資料集

df1 = dataReader.getestdata()

# look at the data

print(df1.head(3))

# check the quality , we could get the missing rate

print(df1.describe())

print("check value!!!")
print(df1['Property_Area'].value_counts())


############################################## EDA

# 確認偏態跟常態等等狀況

