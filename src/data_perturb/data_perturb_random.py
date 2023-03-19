from .data_perturb import CDataPerturb
import numpy as np


class CDataPerturbRandom(CDataPerturb):

    def __init__(self, K=100, min_value=0, max_value=255):
        self.K = K
        self.min_value = min_value
        self.max_value = max_value

    @property
    def K(self):
        return self._K

    @property
    def min_value(self):
        return self._min_value

    @property
    def max_value(self):
        return self._max_value

    @K.setter
    def K(self, value):
        self._K = int(value)

    @min_value.setter
    def min_value(self, value):
        self._min_value = int(value)

    @max_value.setter
    def max_value(self, value):
        self._max_value = int(value)

    def data_perturbation(self, x):
        if x.size != x.shape[0]:
            raise TypeError("x is not flattened!")

        xp = x.copy().ravel()

        idx = np.array(range(0, x.size))
        np.random.shuffle(idx)
        idx = idx[:self.K]

        pixel_values = np.random.randint(self.min_value, self.max_value, self.K)
        xp[idx] = pixel_values
        return xp
