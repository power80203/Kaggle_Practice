import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(".."))
import config
import data.utilities as u

train = pd.read_csv(config.trainset_path)
test = pd.read_csv(config.testset_path)

# check status of datasets

print(train.shape)
print(test.shape)


# set one row of our data to a array
pixes = np.array(test.iloc[1,:])

# reshpae the array to a lsit 28*28
pixes = pixes.reshape(28,28)

#using imshow to print out the figure
plt.imshow(pixes, cmap='gray')
plt.show()


"""
ref: https: // www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6

"""