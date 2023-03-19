import numpy as np
from utils import load_mnist_data
from pandas import read_csv

# x, y = load_mnist_data(
data = read_csv("C:/Users/Utente/Desktop/mnist_data.csv")
print(data.shape)