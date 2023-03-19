from abc import ABC, abstractmethod
import numpy as np


class CDataPerturb(ABC):
    """Abstract interface to implement data perturbation model"""

    @abstractmethod
    def data_perturbation(self, x):
        """

        :param x: a flat vector containing n_features elements
        :return:
        """
        raise NotImplementedError("Data perturbation not implemented!")

    def perturb_dataset(self, x):
        """

        :param x: x is a matrix with shape = (n_samples, n_features)
        :return:
        """
        xp = np.zeros(x.shape)
        for i in range(xp.shape[0]):
            xp[i, :] = self.data_perturbation(x[i, :])
        return xp
