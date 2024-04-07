import unittest

from Classification.Classifier.KMeansClustering import KMeansClustering
from test.Classifier.ClassifierTest import ClassifierTest


class KMeansClusteringTest(ClassifierTest):

    def test_Train(self):
        kMeansCluster = KMeansClustering()
        kMeansCluster.train(self.iris.getInstanceList())
        self.assertAlmostEqual(
            0.0, 100 * kMeansCluster.test(self.iris.getInstanceList()).getErrorRate(), 2
        )
        kMeansCluster.train(self.bupa.getInstanceList())
        self.assertAlmostEqual(
            0.0, 100 * kMeansCluster.test(self.bupa.getInstanceList()).getErrorRate(), 2
        )
        kMeansCluster.train(self.dermatology.getInstanceList())
        self.assertAlmostEqual(
            0.0,
            100 * kMeansCluster.test(self.dermatology.getInstanceList()).getErrorRate(),
            2,
        )
        kMeansCluster.train(self.car.getInstanceList())
        self.assertAlmostEqual(
            0.0, 100 * kMeansCluster.test(self.car.getInstanceList()).getErrorRate(), 2
        )
        kMeansCluster.train(self.tictactoe.getInstanceList())
        self.assertAlmostEqual(
            0.0,
            100 * kMeansCluster.test(self.tictactoe.getInstanceList()).getErrorRate(),
            2,
        )

    def test_Load(self):
        kMeansCluster = KMeansClustering()
        kMeansCluster.loadModel("../../models/bagging-iris.txt")
        self.assertAlmostEqual(
            0.0, 100 * kMeansCluster.test(self.iris.getInstanceList()).getErrorRate(), 2
        )
        kMeansCluster.loadModel("../../models/bagging-bupa.txt")
        self.assertAlmostEqual(
            0.0, 100 * kMeansCluster.test(self.bupa.getInstanceList()).getErrorRate(), 2
        )
        kMeansCluster.loadModel("../../models/bagging-dermatology.txt")
        self.assertAlmostEqual(
            0.0,
            100 * kMeansCluster.test(self.dermatology.getInstanceList()).getErrorRate(),
            2,
        )
        kMeansCluster.loadModel("../../models/bagging-car.txt")
        self.assertAlmostEqual(
            0.0, 100 * kMeansCluster.test(self.car.getInstanceList()).getErrorRate(), 2
        )
        kMeansCluster.loadModel("../../models/bagging-tictactoe.txt")
        self.assertAlmostEqual(
            0.0,
            100 * kMeansCluster.test(self.tictactoe.getInstanceList()).getErrorRate(),
            2,
        )


if __name__ == "__main__":
    unittest.main()
