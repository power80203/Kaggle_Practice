from sklearn.cross_validation import cross_val_score
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris = datasets.load_iris()

print(type(iris))

knn = KNeighborsClassifier(n_neighbors=10)

