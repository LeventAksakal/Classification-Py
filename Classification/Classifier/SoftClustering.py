import math
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Classifier.Clustering import Clustering
from Classification.Instance.Instance import Instance
from Math.Matrix import Matrix
from Util.RandomArray import RandomArray
from Classification.Parameter.Parameter import Parameter


class SoftClustering(Clustering):
    """
    SoftClustering is a class that performs soft clustering using the Expectation-Maximization algorithm.
    It assigns data points to multiple clusters with probabilities indicating the degree of membership.
    """

    def __init__(self, k: int, parameters: Parameter):
        """
        Initializes the SoftClustering object.

        Args:
            k (int): The number of clusters.
            parameters (Parameter): The parameters for the clustering algorithm.
        """
        self.k = k
        self.centroids = InstanceList()
        self.covs = [Matrix(1, 1) for _ in range(self.k)]
        self.weights = [1.0 / self.k for _ in range(self.k)]
        self.parameters = parameters

    def initialize(self, data: InstanceList):
        """
        Initializes the centroids and covariance matrices.

        Args:
            data (InstanceList): The data used for clustering.
        """
        random_indices = RandomArray.indexArray(
            len(data.getInstances()), self.parameters.getSeed()
        )
        self.centroids = InstanceList(
            [data.getInstances()[i] for i in random_indices[: self.k]]
        )
        for i in range(self.k):
            self.covs[i] = data.covariance()

    def e_step(self, data: InstanceList):
        """
        Performs the E-step of the Expectation-Maximization algorithm.

        Args:
            data (InstanceList): The data used for clustering.

        Returns:
            list: A list of cluster probabilities for each data point.
        """
        clusters = [[0 for _ in range(self.k)] for _ in range(len(data.getInstances()))]
        for i, instance in enumerate(data.getInstances()):
            likelihoods = [
                self.weights[j]
                * self.gaussian_pdf(instance, self.centroids.get(j), self.covs[j])
                for j in range(self.k)
            ]
            sum_likelihoods = sum(likelihoods)
            clusters[i] = [likelihood / sum_likelihoods for likelihood in likelihoods]
        return clusters

    def m_step(self, clusters, data: InstanceList):
        """
        Performs the M-step of the Expectation-Maximization algorithm.

        Args:
            clusters (list): A list of cluster probabilities for each data point.
            data (InstanceList): The data used for clustering.
        """
        total_weights = [
            sum(clusters[i][j] for i in range(len(data.getInstances())))
            for j in range(self.k)
        ]
        self.weights = [
            total_weight / len(data.getInstances()) for total_weight in total_weights
        ]
        for j in range(self.k):
            self.centroids.add(
                Instance(
                    sum(
                        clusters[i][j] * data.getInstances()[i].toVector().getValue(0)
                        for i in range(len(data.getInstances()))
                    )
                    / total_weights[j]
                )
            )
            self.covs[j] = self.calculate_covariance(j, clusters, data)

    def gaussian_pdf(self, x, mean, cov):
        """
        Calculates the Gaussian probability density function.

        Args:
            x: The input data point.
            mean: The mean of the Gaussian distribution.
            cov: The covariance matrix of the Gaussian distribution.

        Returns:
            float: The probability density value.
        """
        dim = len(x.toVector())
        x_minus_mean = x.toVector().difference(mean.toVector())
        exp_value = -0.5 * x_minus_mean.transpose().multiply(cov.inverse()).multiply(
            x_minus_mean
        ).getValue(0, 0)
        return (
            1.0 / (math.pow((2.0 * math.pi), dim / 2.0) * math.sqrt(cov.determinant()))
        ) * math.exp(exp_value)

    def calculate_covariance(self, j, clusters, data: InstanceList):
        """
        Calculates the covariance matrix for a specific cluster.

        Args:
            j (int): The index of the cluster.
            clusters (list): A list of cluster probabilities for each data point.
            data (InstanceList): The data used for clustering.

        Returns:
            Matrix: The covariance matrix.
        """
        dim = len(data.getInstances()[0].toVector())
        cov = Matrix(dim, dim)
        for i in range(len(data.getInstances())):
            x_minus_mean = (
                data.getInstances()[i]
                .toVector()
                .difference(self.centroids.get(j).toVector())
            )
            cov = cov.add(
                Matrix(x_minus_mean, dim, 1)
                .multiply(Matrix(x_minus_mean, dim, 1).transpose())
                .multiplyWithConstant(clusters[i][j])
            )
        return cov.multiplyWithConstant(
            1.0 / sum(clusters[i][j] for i in range(len(data.getInstances())))
        )

    def train(self, data, max_iters=100):
        """
        Trains the SoftClustering model using the Expectation-Maximization algorithm.

        Args:
            data (InstanceList): The data used for clustering.
            max_iters (int, optional): The maximum number of iterations. Defaults to 100.

        Returns:
            tuple: A tuple containing the centroids and covariance matrices.
        """
        self.initialize(data)
        for _ in range(max_iters):
            clusters = self.e_step(data)
            self.m_step(clusters, data)
        return self.centroids, self.covs
