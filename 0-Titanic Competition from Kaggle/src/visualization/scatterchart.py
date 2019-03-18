import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from string import ascii_letters
import seaborn as sns
# sns.set(style="white")




def scatterpure(dataset):
    corr = dataset.corr()
    sns.heatmap(corr)
    plt.show()
    


def scatterplot(dataset, var_x, var_y, controlplotshow=True):
    dataset.plot.scatter(x=var_x, y=var_y, title='%s與%s成績關聯性' % (var_x, var_y))
    plt.savefig('%s%s與%s關聯性.png' % (abs_file_path, var_x, var_y),
                bbox_inches='tight', pad_inches=0.5)
    if controlplotshow:
        plt.show()
    else:
        plt.close()

