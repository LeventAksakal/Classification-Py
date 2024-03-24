from Classification.StatisticalTest.KruskalWallis import KruskalWallis
import unittest
from Classification.Classifier.Qda import QdaModel
from Classification.Classifier.C45 import C45
from Classification.Classifier.Knn import Knn
from Classification.Classifier.Lda import Lda
from Classification.Classifier.NaiveBayes import NaiveBayes
from Classification.DistanceMetric.EuclidianDistance import EuclidianDistance
from Classification.Experiment.Experiment import Experiment
from Classification.Experiment.KFoldRun import KFoldRun
from Classification.Parameter.C45Parameter import C45Parameter
from Classification.Parameter.KnnParameter import KnnParameter
from Classification.Parameter.Parameter import Parameter
from test.Classifier.ClassifierTest import ClassifierTest


class KruskalWallisTest(ClassifierTest):
    """
    Unit tests for the Kruskal-Wallis statistical test.

    These tests verify the functionality of the Kruskal-Wallis statistical test by comparing experiment performances.

    Attributes:
        self.bupa: The dataset used for testing with Bupa.
        self.iris: The dataset used for testing with Iris.
        self.dermatology: The dataset used for testing with Dermatology.
    """

    def test_Compare(self):
        """
        Test the compare method of Kruskal-Wallis.

        This method performs unit tests on the compare method of Kruskal-Wallis by comparing experiment performances.
        """
        kFoldRun = KFoldRun(10)
        kruskalWallis = KruskalWallis()
        experimentPerformance1 = kFoldRun.execute(
            Experiment(C45(), C45Parameter(1, True, 0.2), self.bupa)
        )
        experimentPerformance2 = kFoldRun.execute(
            Experiment(
                Knn(),
                KnnParameter(1, 10, EuclidianDistance()),
                self.bupa,
            )
        )
        experimentPerformance3 = kFoldRun.execute(
            Experiment(Lda(), Parameter(1), self.bupa)
        )
        self.assertAlmostEqual(
            1,
            kruskalWallis.compare(
                experimentPerformance1, experimentPerformance2, experimentPerformance3
            ),
            delta=0.5,
        )
        experimentPerformance1 = kFoldRun.execute(
            Experiment(C45(), C45Parameter(1, True, 0.2), self.iris)
        )
        experimentPerformance2 = kFoldRun.execute(
            Experiment(Knn(), KnnParameter(1, 10, EuclidianDistance()), self.iris)
        )
        experimentPerformance3 = kFoldRun.execute(
            Experiment(Lda(), Parameter(1), self.iris)
        )
        self.assertAlmostEqual(
            1,
            kruskalWallis.compare(
                experimentPerformance1, experimentPerformance2, experimentPerformance3
            ),
            delta=0.5,
        )
        experimentPerformance1 = kFoldRun.execute(
            Experiment(C45(), C45Parameter(1, True, 0.2), self.dermatology)
        )
        experimentPerformance2 = kFoldRun.execute(
            Experiment(
                Knn(), KnnParameter(1, 10, EuclidianDistance()), self.dermatology
            )
        )
        experimentPerformance3 = kFoldRun.execute(
            Experiment(NaiveBayes(), Parameter(1), self.dermatology)
        )
        self.assertAlmostEqual(
            1,
            kruskalWallis.compare(
                experimentPerformance1, experimentPerformance2, experimentPerformance3
            ).getPValue(),
            delta=0.5,
        )
        experimentPerformance1 = kFoldRun.execute(
            Experiment(C45(), C45Parameter(1, True, 0.2), self.iris)
        )
        experimentPerformance2 = kFoldRun.execute(
            Experiment(Lda(), Parameter(1), self.iris)
        )
        experimentPerformance3 = kFoldRun.execute(
            Experiment(QdaModel(), Parameter(1), self.iris)
        )
        self.assertAlmostEqual(
            1,
            kruskalWallis.compare(
                experimentPerformance1, experimentPerformance2, experimentPerformance3
            ),
            delta=0.5,
        )


if __name__ == "__main__":
    unittest.main()
