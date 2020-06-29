import unittest

from Classification.Classifier.C45 import C45
from Classification.Parameter.C45Parameter import C45Parameter
from test.Classifier.ClassifierTest import ClassifierTest


class C45Test(ClassifierTest):

    def test_Train(self):
        c45 = C45()
        c45Parameter = C45Parameter(1, True, 0.2)
        c45.train(self.iris.getInstanceList(), c45Parameter)
        self.assertAlmostEqual(2.67, 100 * c45.test(self.iris.getInstanceList()).getErrorRate(), 2)
        c45.train(self.bupa.getInstanceList(), c45Parameter)
        self.assertAlmostEqual(42.03, 100 * c45.test(self.bupa.getInstanceList()).getErrorRate(), 2)
        c45.train(self.dermatology.getInstanceList(), c45Parameter)
        self.assertAlmostEqual(1.91, 100 * c45.test(self.dermatology.getInstanceList()).getErrorRate(), 2)
        c45.train(self.car.getInstanceList(), c45Parameter)
        self.assertAlmostEqual(1.16, 100 * c45.test(self.car.getInstanceList()).getErrorRate(), 2)
        c45.train(self.tictactoe.getInstanceList(), c45Parameter)
        self.assertAlmostEqual(1.77, 100 * c45.test(self.tictactoe.getInstanceList()).getErrorRate(), 2)
        c45.train(self.nursery.getInstanceList(), c45Parameter)
        self.assertAlmostEqual(0.30, 100 * c45.test(self.nursery.getInstanceList()).getErrorRate(), 2)
        c45.train(self.chess.getInstanceList(), c45Parameter)
        self.assertAlmostEqual(47.61, 100 * c45.test(self.chess.getInstanceList()).getErrorRate(), 2)


if __name__ == '__main__':
    unittest.main()
