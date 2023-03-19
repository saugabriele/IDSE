from sklearn.metrics import pairwise_distances
import numpy as np

class NMC:
    # Class implementing the nearest mean centroids classification algorithm!
    def __init__(self, robust_centroid_estimation=False):
        #print("object created")
        """
        Create the NMC object.
        :param robust_centroid_estimation: True use median to compute the centroids, False use mean
        """
        self._robust_centroid_estimation = robust_centroid_estimation  # Avg true means mean and false means median
        self._centroids = None  # init centroids

    @property
    def centroids(self):
        return self._centroids

    @property
    def robust_centroid_estimation(self):
        return self._robust_centroid_estimation

    @robust_centroid_estimation.setter
    def robust_centroid_estimation(self, value):
        if not isinstance(value, bool):
            raise TypeError()
        self._robust_centroid_estimation = value

    def fit(self, x_tr, y_tr):
        # Fit the model to the data
        n_classes = np.unique(y_tr).size
        n_features = x_tr.shape[1]
        self._centroids = np.zeros(shape=(n_classes, n_features))
        for k in range(n_classes):
            # extract only image of 0 from x_tr
            xk = x_tr[y_tr == k, :]
            if self._robust_centroid_estimation is True:
                self._centroids[k, :] = np.median(xk, axis=0)
            else:
                self._centroids[k, :] = np.mean(xk, axis=0)
        return self

    def predict(self, x_ts):
        dist = pairwise_distances(x_ts, self._centroids)
        y_pred = np.argmin(dist, axis=1)
        return y_pred