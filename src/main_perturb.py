import matplotlib.pyplot as plt
import numpy as np
from utils import *
from sklearn.model_selection import ShuffleSplit
from nmc import NMC

from data_perturb import CDataPerturbRandom

data_pert = CDataPerturbRandom()

x, y = load_mnist_data()
xp = data_pert.perturb_dataset(x)
# plot_ten_digits(x, y)
# plot_ten_digits(xp, y)

# Split MNIST data in 60% training and 40% test set
n_tr = round(0.6 * x.shape[0])
x_tr, y_tr, x_ts, y_ts = split_data(x, y, n_tr=n_tr)

clf = NMC()
clf.fit(x_tr, y_tr)
y_pred = clf.predict(x_ts)

clf_acc = np.mean(y_ts == y_pred)
print("Test accuracy: ", int(clf_acc*1000)/10, "%")

k_values = np.array([0, 10, 20, 50, 100, 200, 400, 500])
test_accuracies = np.zeros(shape=k_values.shape)
for i, k in enumerate(k_values):
    # perturb test set
    data_pert.K = k
    xp = data_pert.perturb_dataset(x_ts)
    # Compute predicted labels on the perturbed test set
    y_pred = clf.predict(xp)
    # Compute classification accuracy on the predicted labels
    clf_acc = np.mean(y_ts == y_pred)
    print("Test accuracy(K=", k,"): ", int(clf_acc * 1000) / 10, "%")
    test_accuracies[i] = clf_acc

plt.plot(k_values, test_accuracies)
plt.show()