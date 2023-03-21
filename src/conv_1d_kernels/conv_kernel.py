from abc import ABC, abstractmethod


class CConvKernel(ABC):

    def __init__(self, kernel_size=3):
        self._mask = None
        self.kernel_size = kernel_size

    @property
    def mask(self):
        return self._mask

    @property
    def kernel_size(self):
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, value):
        # todo: check if the value is an odd number!
        if value % 2 == 0:
            raise ValueError("value ii not an odd number")
        self._kernel_size = int(value)
        self.kernel_mask()  # This gonna be called from the child

    @abstractmethod
    def kernel_mask(self):
        raise NotImplementedError()

    def kernel(self, x):
        xp = x.copy()
        offset = (self.mask.size - 1) // 2
        for i in range(x.size - (self.mask.size - 1)):
            xp[i + offset] = self.mask.dot(x[i:i + self.mask.size])
        return xp
