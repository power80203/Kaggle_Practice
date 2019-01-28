import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
p = Path(__file__).parents[2]
sys.path.append("%s"%p)
import src.preparation.config as config

df_test = pd.read_csv(config.traindatapath)

def getestdata():
    return(df_test)



if __name__ == '__main__':
    try:
        print(df_test.head(3))
    except Exception:
        print("沒成功啦")
    


