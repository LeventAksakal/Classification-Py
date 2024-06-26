from Classification.Classifier.Clustering import Clustering
from Classification.InstanceList.InstanceList import InstanceList
from Classification.DistanceMetric.EuclidianDistance import EuclidianDistance


class KMeansClustering(Clustering):
    """
    KMeansClustering class that extends Clustering.

    Attributes:
        k (int): Number of clusters.
        centroids (InstanceList): List of centroids.
    """

    def __init__(self, k):
        """
        Constructor of KMeansClustering class.

        Parameters:
            k (int): Number of clusters.
        """
        self.k = k
        self.centroids = InstanceList()

    def assign_points_to_clusters(self, centroids, data):
        """
        Assigns each data point to the closest centroid.

        Parameters:
            centroids (InstanceList): List of centroids.
            data (InstanceList): List of data points.

        Returns:
            list: List of clusters.
        """
        clusters = [InstanceList() for _ in range(self.k)]
        for instance in data.getInstances():
            closest_centroid_index = min(
                range(self.k),
                key=lambda index: EuclidianDistance().distance(
                    instance, centroids.get(index)
                ),
            )
            clusters[closest_centroid_index].add(instance)
        return clusters

    def calculate_new_centroids(self, clusters):
        """
        Calculates new centroids as the mean of each cluster.

        Parameters:
            clusters (list): List of clusters.

        Returns:
            list: List of new centroids.
        """
        return [cluster.meanInstance() for cluster in clusters]

    def train(self, trainSet: InstanceList):
        """
        Trains the KMeans clustering model.

        Parameters:
            trainSet (InstanceList): Training data.

        Returns:
            list: List of final clusters.
        """
        self.centroids = self.calculate_new_centroids(
            [trainSet.subList(i, i + 1) for i in range(self.k)]
        )

        while True:
            clusters = self.assign_points_to_clusters(self.centroids, trainSet)
            new_centroids = self.calculate_new_centroids(clusters)

            if self.centroids == new_centroids:
                break

            self.centroids = new_centroids

        return clusters
