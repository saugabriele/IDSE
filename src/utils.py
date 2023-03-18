import numpy as np
import pandas
from sklearn.datasets import fetch_openml


def load_mnist_data():
    mnist = fetch_openml('mnist_784', cache=True)
    x = np.array(mnist.data / 255)
    y = np.array(mnist.target, dtype=int)
    return x, y