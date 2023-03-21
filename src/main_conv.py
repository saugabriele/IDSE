import numpy as np
from utils import *
import matplotlib.pyplot as plt

from conv_1d_kernels import CConvKernelMovingAverage, CConvKernelTriangle

x, y = load_mnist_data()

z = x[0, :]

# conv = CConvKernelMovingAverage()
conv = CConvKernelTriangle(kernel_size=15)
print(conv.mask)

zp = conv.kernel(z)

plt.imshow(zp.reshape(28,28))
plt.show()
