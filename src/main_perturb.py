import matplotlib.pyplot as plt
import numpy as np
from utils import *
from sklearn.model_selection import ShuffleSplit

from data_perturb import CDataPerturbRandom

data_pert = CDataPerturbRandom()

x, y = load_mnist_data()

xp = data_pert.perturb_dataset(x)

plot_ten_digits(x, y)
plot_ten_digits(xp)
