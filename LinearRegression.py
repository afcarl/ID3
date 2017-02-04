import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import Quandl as quandl


df = quandl.get("GOOG/NASDAQ_GOOGL", authtoken="seb_UHjiaf6bEKswahhz")


# Division of attributes to X and Y

y = df['Close'][:-1]
x = df[['Open', 'High', 'Low', 'Volume']][:-1]

# Using Linear Regression


def compute_pseudo_inverse(x):
    x_transpose = x.transpose()
    x = x_transpose.dot(x)
    x_inv = pd.DataFrame(np.linalg.pinv(x.values), x.columns, x.index)
    df = x_inv.dot(x_transpose)
    return df


def compute_weights(x,y):
    w = np.dot(x,y)
    return w


w = compute_weights(compute_pseudo_inverse(x),y)

# Check the value for the last element
x_t = x.tail(1).transpose()
y_predicted = np.dot(w, x_t)

print(y_predicted, y[-1])







