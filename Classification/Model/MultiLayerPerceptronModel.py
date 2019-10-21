from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.LinearPerceptronModel import LinearPerceptronModel
from Classification.Parameter.MultiLayerPerceptronParameter import MultiLayerPerceptronParameter
import copy

from Classification.Performance.ClassificationPerformance import ClassificationPerformance


class MultiLayerPerceptronModel(LinearPerceptronModel):

    """
    The allocateWeights method allocates layers' weights of Matrix W and V.

    PARAMETERS
    ----------
    H : int
        Integer value for weights.
    """
    def allocateWeights(self, H: int):
        self.W = self.allocateLayerWeights(H, self.d + 1)
        self.V = self.allocateLayerWeights(self.K, H + 1)

    """
    A constructor that takes InstanceLists as trainsSet and validationSet. It  sets the NeuralNetworkModel nodes with 
    given InstanceList then creates an input vector by using given trainSet and finds error. Via the validationSet it 
    finds the classification performance and reassigns the allocated weight Matrix with the matrix that has the best 
    accuracy and the Matrix V with the best Vector input.

    PARAMETERS
    ----------
    trainSet : InstanceList     
        InstanceList that is used to train.
    validationSet : InstanceList
        InstanceList that is used to validate.
    parameters : MultiLayerPerceptronParameter   
        Multi layer perceptron parameters; seed, learningRate, etaDecrease, crossValidationRatio, epoch, hiddenNodes.
    """
    def __init__(self, trainSet: InstanceList, validationSet: InstanceList, parameters: MultiLayerPerceptronParameter):
        super().initWithTrainSet(trainSet)
        self.allocateWeights(parameters.getHiddenNodes())
        bestW = copy.deepcopy(self.W)
        bestV = copy.deepcopy(self.V)
        bestClassificationPerformance = ClassificationPerformance(0.0)
        epoch = parameters.getEpoch()
        learningRate = parameters.getLearningRate()
        for i in range(epoch):
            trainSet.shuffle(parameters.getSeed())
            for j in range(trainSet.size()):
                self.createInputVector(trainSet.get(j))
                hidden = self.calculateHidden(self.x, self.W)
                hiddenBiased = hidden.biased()
                rMinusY = self.calculateRMinusY(trainSet.get(j), hiddenBiased, self.V)
                deltaV = rMinusY.multiplyWithVector(hiddenBiased)
                oneMinusHidden = self.calculateOneMinusHidden(hidden)
                tmph = self.V.multiplyWithVectorFromLeft(rMinusY)
                tmph.remove(0)
                tmpHidden = oneMinusHidden.elementProduct(hidden.elementProduct(tmph))
                deltaW = tmpHidden.multiplyWithVector(self.x)
                deltaV.multiplyWithConstant(learningRate)
                self.V.add(deltaV)
                deltaW.multiplyWithConstant(learningRate)
                self.W.add(deltaW)
            currentClassificationPerformance = self.testClassifier(validationSet)
            if currentClassificationPerformance.getAccuracy() > bestClassificationPerformance.getAccuracy():
                bestClassificationPerformance = currentClassificationPerformance
                bestW = copy.deepcopy(self.W)
                bestV = copy.deepcopy(self.V)
            learningRate *= parameters.getEtaDecrease()
        self.W = bestW
        self.V = bestV