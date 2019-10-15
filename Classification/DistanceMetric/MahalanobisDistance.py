from Math.Matrix import Matrix

from Classification.DistanceMetric.DistanceMetric import DistanceMetric
from Classification.Instance.Instance import Instance


class MahalanobisDistance(DistanceMetric):

    """
    Constructor for the MahalanobisDistance class. Basically sets the inverse of the covariance matrix.

    PARAMETERS
    ----------
    covarianceInverse : Matrix
        Inverse of the covariance matrix.
    """
    def __init__(self, covarianceInverse: Matrix):
        self.covarianceInverse = covarianceInverse

    """
    Calculates Mahalanobis distance between two instances. (x^(1) - x^(2)) S (x^(1) - x^(2))^T

    PARAMETERS
    ----------
    instance1 : Instance
        First instance.
    instance2 : Instance
        Second instance.
    
    RETURNS
    -------
    float
        Mahalanobis distance between two instances.
    """
    def distance(self, instance1: Instance, instance2: Instance) -> float:
        v1 = instance1.toVector()
        v2 = instance2.toVector()
        v1.subtract(v2)
        v3 = self.covarianceInverse.multiplyWithVectorFromLeft(v1)
        return v3.dotProduct(v1)