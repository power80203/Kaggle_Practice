import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import sys


def barchart(dataset, var_x, var_y, topf=None, controlplotshow=True):
    """

    """
    df_new = dataset.groupby(by=var_x)[var_y].count()
    df_new.sort_values(ascending=True)[:topf].plot(
        kind='bar', title='%s' % var_x, rot=0)
    plt.xlabel(var_x)
    plt.ylabel(var_y)

    for counter, value in enumerate(df_new.sort_values(ascending=True)[:topf]):
        plt.text(counter, value, s="%.2f" %
                 value, ha='center', va='bottom', fontsize=10)

    # plt.savefig('%s%s之平均%s.png' % (abs_file_path, var_x, var_y),
    #             bbox_inches='tight', pad_inches=0.5)
    #plt.savefig('Hualien Project-Python\\reports\\figures\\bar_plot\\%s之平均%s.png'%(var_x,var_y), \
    #bbox_inches = 'tight', pad_inches = 0.5)
    if controlplotshow:
        plt.show()
    else:
        plt.close()


def barchart_onevar(dataset, var_x ,qc = False, y_title = "y", controlplotshow = True):
    if qc:
        df_new = dataset.query(qc)[var_x].value_counts().sort_index()
    else:
        df_new = dataset[var_x].value_counts().sort_index()
    df_new.plot(kind='bar', title='%s之%s' % (var_x, y_title), rot=0)
    plt.xlabel(var_x)
    plt.ylabel(y_title)
    if controlplotshow:
        plt.show()
    else:
        plt.close()


def histchart_for_continuous(dataset, var_x, y_title, controlplotshow=True):
    dataset[var_x].plot.hist(title='%s之%s' % (var_x, y_title))
    plt.xlabel(var_x)
    plt.ylabel(y_title)
    plt.savefig('%s%s之%s.png' % (abs_file_path, var_x, y_title),
                bbox_inches='tight', pad_inches=0.5)
    if controlplotshow:
        plt.show()
    else:
        plt.close()
