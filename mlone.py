import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import Quandl

from sklearn import svm

digits = datasets.load_digits()
# Setup our classifier
clf = svm.SVC(gamma=0.001, C=100)

x, y = digits.data[:-1], digits.target[:-1]

clf.fit(x,y)

print("Prediction is", clf.predict(digits.data[-1]))

plt.imshow(digits.images[-1])
