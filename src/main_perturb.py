import matplotlib.pyplot as plt
import numpy as np
from utils import *
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC
from nmc import NMC

from data_perturb import CDataPerturbRandom, CDataPerturbGaussian


def robustness_test(clf, data_pert, param_name, param_values):
    test_accuracies = np.zeros(shape=param_values.shape)
    for i, k in enumerate(param_values):
        # perturb test set
        setattr(data_pert, param_name, k)
        xp = data_pert.perturb_dataset(x_ts)
        # Compute predicted labels on the perturbed test set
        y_pred = clf.predict(xp)
        # Compute classification accuracy on the predicted labels
        clf_acc = np.mean(y_ts == y_pred)
        #print("Test accuracy(K=", k, "): ", int(clf_acc * 1000) / 10, "%")
        test_accuracies[i] = clf_acc
    return test_accuracies


data_pert = CDataPerturbGaussian()  # CDataPerturbRandom()

x, y = load_mnist_data()
xp = data_pert.perturb_dataset(x)
# plot_ten_digits(x, y)
# plot_ten_digits(xp, y)

# Split MNIST data in 60% training and 40% test set
n_tr = round(0.6 * x.shape[0])
x_tr, y_tr, x_ts, y_ts = split_data(x, y, n_tr=n_tr)

param_values = np.array([0, 10, 20, 50, 100, 200, 400, 500])

clf_list = [NMC(), SVC(kernel='linear')]
clf_names = ['NMC', 'SVM']

plt.figure(figsize=(10, 5))

for i, clf in enumerate(clf_list):
    clf.fit(x_tr, y_tr)
    y_pred = clf.predict(x_ts)
    clf_acc = np.mean(y_ts == y_pred)
    print("Test accuracy: ", int(clf_acc * 1000) / 10, "%")

    test_accuracies = robustness_test(clf, CDataPerturbRandom(), param_name='K', param_values=param_values)

    plt.subplot(1, 2, 1)
    plt.plot(param_values, test_accuracies, label=clf_names[i])
    plt.xlabel('X')
    plt.ylabel("Test accuracy(K)")
    plt.legend()

    test_accuracies = robustness_test(clf, CDataPerturbGaussian(), param_name='sigma', param_values=param_values)

    plt.subplot(1, 2, 2)
    plt.plot(param_values, test_accuracies, label=clf_names[i])
    plt.xlabel(r'$\Sigma$')
    plt.ylabel(r'Test accuracies($\sigma$)')
    plt.legend()
plt.show()
