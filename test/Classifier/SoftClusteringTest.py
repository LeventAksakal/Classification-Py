import unittest

from Classification.Classifier.SoftClustering import SoftClustering
from test.Classifier.ClassifierTest import ClassifierTest


class SoftClusteringTest(ClassifierTest):

    def test_Train(self):
        softCluster = SoftClustering()
        softCluster.train(self.iris.getInstanceList())
        self.assertAlmostEqual(
            0.0, 100 * softCluster.test(self.iris.getInstanceList()).getErrorRate(), 2
        )
        softCluster.train(self.bupa.getInstanceList())
        self.assertAlmostEqual(
            0.0, 100 * softCluster.test(self.bupa.getInstanceList()).getErrorRate(), 2
        )
        softCluster.train(self.dermatology.getInstanceList())
        self.assertAlmostEqual(
            0.0,
            100 * softCluster.test(self.dermatology.getInstanceList()).getErrorRate(),
            2,
        )
        softCluster.train(self.car.getInstanceList())
        self.assertAlmostEqual(
            0.0, 100 * softCluster.test(self.car.getInstanceList()).getErrorRate(), 2
        )
        softCluster.train(self.tictactoe.getInstanceList())
        self.assertAlmostEqual(
            0.0,
            100 * softCluster.test(self.tictactoe.getInstanceList()).getErrorRate(),
            2,
        )

    def test_Load(self):
        softCluster = SoftClustering()
        softCluster.loadModel("../../models/bagging-iris.txt")
        self.assertAlmostEqual(
            0.0, 100 * softCluster.test(self.iris.getInstanceList()).getErrorRate(), 2
        )
        softCluster.loadModel("../../models/bagging-bupa.txt")
        self.assertAlmostEqual(
            0.0, 100 * softCluster.test(self.bupa.getInstanceList()).getErrorRate(), 2
        )
        softCluster.loadModel("../../models/bagging-dermatology.txt")
        self.assertAlmostEqual(
            0.0,
            100 * softCluster.test(self.dermatology.getInstanceList()).getErrorRate(),
            2,
        )
        softCluster.loadModel("../../models/bagging-car.txt")
        self.assertAlmostEqual(
            0.0, 100 * softCluster.test(self.car.getInstanceList()).getErrorRate(), 2
        )
        softCluster.loadModel("../../models/bagging-tictactoe.txt")
        self.assertAlmostEqual(
            0.0,
            100 * softCluster.test(self.tictactoe.getInstanceList()).getErrorRate(),
            2,
        )


if __name__ == "__main__":
    unittest.main()
