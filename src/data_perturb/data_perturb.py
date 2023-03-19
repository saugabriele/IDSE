from abc import ABC, abstractmethod
import numpy as np


class CDataPerturb(ABC):
    """Abstract interface to implement data perturbation model"""

    @abstractmethod
    def data_perturbation(self, x):
        """

        Parameters
        ----------
        x: ndarray flattened vector to be perturbed

        Returns
        -------
        xp: the perturbed version of x
        """
        raise NotImplementedError("Data perturbation not implemented!")

    def perturb_dataset(self, x):
        """

        Parameters
        ----------
        x: x is a matrix with shape = (n_samples, n_features)

        Returns
        -------
        xp: the perturbed version of x
        """
        xp = np.zeros(x.shape)
        for i in range(xp.shape[0]):
            xp[i, :] = self.data_perturbation(x[i, :])
        return xp
