import unittest

from Classification.Classifier.Bagging import Bagging
from Classification.Classifier.C45 import C45
from Classification.Classifier.Dummy import Dummy
from Classification.Classifier.Knn import Knn
from Classification.Classifier.Lda import Lda
from Classification.Classifier.LinearPerceptron import LinearPerceptron
from Classification.Classifier.NaiveBayes import NaiveBayes
from Classification.DistanceMetric.EuclidianDistance import EuclidianDistance
from Classification.Experiment.BootstrapRun import BootstrapRun
from Classification.Experiment.Experiment import Experiment
from Classification.Parameter.BaggingParameter import BaggingParameter
from Classification.Parameter.C45Parameter import C45Parameter
from Classification.Parameter.KnnParameter import KnnParameter
from Classification.Parameter.LinearPerceptronParameter import LinearPerceptronParameter
from Classification.Parameter.Parameter import Parameter
from test.Classifier.ClassifierTest import ClassifierTest


class BootstrapRunTest(ClassifierTest):

    def test_Execute(self):
        bootstrapRun = BootstrapRun(10)
        experimentPerformance = bootstrapRun.execute(Experiment(C45(), C45Parameter(1, True, 0.2), self.iris))
        self.assertAlmostEqual(3.8, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = bootstrapRun.execute(Experiment(C45(), C45Parameter(1, True, 0.2), self.tictactoe))
        self.assertAlmostEqual(7.28, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = bootstrapRun.execute(Experiment(Knn(), KnnParameter(1, 3, EuclidianDistance()), self.bupa))
        self.assertAlmostEqual(24.84, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = bootstrapRun.execute(Experiment(Knn(), KnnParameter(1, 3, EuclidianDistance()), self.dermatology))
        self.assertAlmostEqual(8.43, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = bootstrapRun.execute(Experiment(Lda(), Parameter(1), self.bupa))
        self.assertAlmostEqual(32.03, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = bootstrapRun.execute(Experiment(Lda(), Parameter(1), self.dermatology))
        self.assertAlmostEqual(2.95, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = bootstrapRun.execute(Experiment(LinearPerceptron(), LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), self.iris))
        self.assertAlmostEqual(3.26, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = bootstrapRun.execute(Experiment(LinearPerceptron(), LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100), self.dermatology))
        self.assertAlmostEqual(2.65, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = bootstrapRun.execute(Experiment(NaiveBayes(), Parameter(1), self.car))
        self.assertAlmostEqual(14.75, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = bootstrapRun.execute(Experiment(NaiveBayes(), Parameter(1), self.nursery))
        self.assertAlmostEqual(9.71, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = bootstrapRun.execute(Experiment(Bagging(), BaggingParameter(1, 50), self.tictactoe))
        self.assertAlmostEqual(3.00, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = bootstrapRun.execute(Experiment(Bagging(), BaggingParameter(1, 50), self.car))
        self.assertAlmostEqual(3.44, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = bootstrapRun.execute(Experiment(Dummy(), Parameter(1), self.nursery))
        self.assertAlmostEqual(66.79, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)
        experimentPerformance = bootstrapRun.execute(Experiment(Dummy(), Parameter(1), self.iris))
        self.assertAlmostEqual(66.67, 100 * experimentPerformance.meanPerformance().getErrorRate(), 2)


if __name__ == '__main__':
    unittest.main()
