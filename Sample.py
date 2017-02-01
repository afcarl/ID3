import numpy as np

# Data initialization
o = np.array([0,0,1,2,2,2,1,0,0,2,0,1,1,2])
h = np.array([0,0,0,0,1,1,1,0,1,1,1,0,1,0])
t = np.array([0,0,0,1,2,2,2,1,2,1,1,1,0,1])
w = np.array([0,1,0,0,0,1,1,0,0,0,1,1,0,1])
y = np.array([0,0,1,1,1,0,1,0,1,1,1,1,1,0])

def partition(a):
    return {c: (a==c).nonzero()[0] for c in np.unique(a)}

def entropy(s):
    res = 0
    val, counts = np.unique(s, return_counts=True)
    freqs = counts.astype('float') / len(s)
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res


def mutual_information(y, x):

    # res = -p+log2(p+)-(p-log2(p-))
    res = entropy(y)

    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float')/len(x)

    # Calculate the weighted average
    # Formula for the same is Gain = res -  sumforvaluesofA(p*entropy(value of A))
    # Use zip to combine the values of both the freqs list and val list
    for p, v in zip(freqs, val):
        res -= p * entropy(y[x == v])

    return res

def is_pure(s):
    return len(set(s)) == 1

def split_data(x, y):

    if is_pure(y) or len(y) == 0:
        return y

    gain = np.array([mutual_information(y, x_attr) for x_attr in x.T])
    selected_attr = np.argmax(gain)


    if np.all(gain < 1e-6):
        return y

    # Split using the seleted attribute
    split_array = x[:,selected_attr]
    sets = partition(split_array)


    # Define a result dictionary
    res = {}
    for k, v in sets.items():
        y_sub = y.take(v,axis=0)
        x_sub = x.take(v,axis=0)

        res["x%d = %d" %(selected_attr,k)] = split_data(x_sub,y_sub)

    return res

d = np.array([o,h,t,w]).T
print(split_data(d,y))

