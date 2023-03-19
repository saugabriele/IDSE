import numpy as np
from utils import *
from nmc import NMC

n_rep = 5
x, y = load_mnist_data()

test_error = np.zeros(shape=(n_rep,))
for i in range(n_rep):
    x_tr, y_tr, x_ts, y_ts = split_data(x, y, n_tr=1000)
    clf = NMC(robust_centroid_estimation=True)
    clf.fit(x_tr, y_tr)
    # plot_ten_digits(clf.centroids)
    ypred = clf.predict(x_ts)
    test_error[i] = (ypred != y_ts).mean()

print(test_error.mean(), test_error.std())  # Test error and standard deviation
