from Classification.Classifier.Clustering import Clustering
from Classification.InstanceList.InstanceList import InstanceList
from Classification.DistanceMetric.EuclidianDistance import EuclidianDistance


class KMeansClustering(Clustering):
    def __init__(self, k):
        self.k = k
        self.centroids = InstanceList()

    def assign_points_to_clusters(self, centroids, data):
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
        return [cluster.meanInstance() for cluster in clusters]

    def train(self, trainSet: InstanceList):
        n_features = trainSet.get(0).attributeSize()

        # Randomly sample k data points and calculate their average to initialize the centroids
        self.centroids = self.calculate_new_centroids(
            [trainSet.subList(i, i + 1) for i in range(self.k)]
        )

        while True:
            clusters = self.assign_points_to_clusters(self.centroids, trainSet)
            new_centroids = self.calculate_new_centroids(clusters)

            if self.centroids == new_centroids:
                break

            self.centroids = new_centroids

        return clusters  # Return the clusters
