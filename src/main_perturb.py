import matplotlib.pyplot as plt
import numpy as np
from utils import *
from sklearn.model_selection import ShuffleSplit

from data_perturb import CDataPerturbRandom

data_pert = CDataPerturbRandom()

x, y = load_mnist_data()

xp = data_pert.perturb_dataset(x)

plt.imshow(xp[0, :].reshape(28,28))
plt.show()
plt.imshow(xp[1, :].reshape(28,28))
plt.show()

plt.imshow(x[0, :].reshape(28,28))
plt.show()
plt.imshow(x[1, :].reshape(28,28))
plt.show()
