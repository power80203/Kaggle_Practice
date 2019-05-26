import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
#!/usr/bin/env python3
import torch

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import models
from keras import layers
from keras import optimizers
# load models
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score


sns.set(style='white', context='notebook', palette='deep')

sys.path.append(os.path.abspath(".."))

import config
import features.featureEngineering as FE



dataset, train_len, IDtest = FE.featureEngineeringMain()
train = dataset[:train_len] # 以前 是 train
test = dataset[train_len:] # 以後是 test
test.drop(labels=["Survived"],axis = 1,inplace=True)

#########################################################
# Separate train features and label 

train["Survived"] = train["Survived"].astype(int)

Y_train = train["Survived"]
Y_train = torch.from_numpy(Y_train.values)

X_train = train.drop(labels = ["Survived"],axis = 1)
X_train = torch.from_numpy(X_train.values)
X_train = X_train.type(torch.FloatTensor)

print(Y_train.shape)
print(X_train.shape)



def nn_torch(x,y):
    # Code in file nn/two_layer_net_optim.py
    import torch

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    print( X_train.shape[1])
    N, D_in,D_out = 100, X_train.shape[1], 1

    # Create random Tensors to hold inputs and outputs.


    # Use the nn package to define our model and loss function.
    model = torch.nn.Sequential(
            torch.nn.Linear(D_in, 128),
            torch.nn.ReLU(32),
            torch.nn.Sigmoid(),
            )
    loss_fn = torch.nn.CrossEntropyLoss()

    # Use the optim package to define an Optimizer that will update the weights of
    # the model for us. Here we will use Adam; the optim package contains many other
    # optimization algoriths. The first argument to the Adam constructor tells the
    # optimizer which Tensors it should update.
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(model)

    for t in range(100):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(X_train)
        # Compute and print loss.
        loss = loss_fn(y_pred, Y_train)
        print(t, loss.item())
    
        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the Tensors it will update (which are the learnable weights
        # of the model)
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()

    
    y_pred_2 = model(x)
    y_pred_2 = y_pred_2.detach().numpy()
    # y_pred_2 = y_pred_2.detach().numpy()
    
    y_pred_2 = y_pred_2 > 0.5

    # y_pred_2 = [int(1) if i > 0.5 else int(0) for i in y_pred_2]


    # y_pred_2 = [int(1) if i == True else int(0) for i in y_pred_2]
    # print(y_pred_2)
    # print(Y_train)

    print(type(y_pred_2))
    print(type(Y_train))

    sum = 0
    for i in range(len(y_pred_2)):
        print(y_pred_2[i], Y_train[i])
        print("====")
        if y_pred_2[i] == Y_train[i]:
            sum +=1
            print(sum)
    


    acc = sum/len(y_pred_2)
    
    

    # acc = accuracy_score(y_pred_2, Y_train)



    print("accurancy in testset is {}".format(acc))







if __name__ == "__main__":
    nn_torch(X_train, Y_train)