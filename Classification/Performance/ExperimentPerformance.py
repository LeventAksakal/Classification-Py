from __future__ import annotations
from Classification.Performance.Performance import Performance
from Classification.Performance.ClassificationPerformance import (
    ClassificationPerformance,
)
from Classification.Performance.DetailedClassificationPerformance import (
    DetailedClassificationPerformance,
)
import math


class ExperimentPerformance:

    __results: list
    __contains_details: bool
    __classification: bool

    def __init__(self):
        """
        A constructor which creates a new list of Performance as results.
        """
        self.__results = []
        self.__contains_details = True
        self.__classification = True

    def __gt__(self, other) -> bool:
        accuracy1 = self.meanClassificationPerformance().getAccuracy()
        accuracy2 = other.meanClassificationPerformance().getAccuracy()
        return accuracy1 > accuracy2

    def __lt__(self, other) -> bool:
        accuracy1 = self.meanClassificationPerformance().getAccuracy()
        accuracy2 = other.meanClassificationPerformance().getAccuracy()
        return accuracy1 < accuracy2

    def initWithFile(self, fileName: str):
        """
        A constructor that takes a file name as an input and takes the inputs from that file assigns these inputs to the
        errorRate and adds them to the results list as a new Performance.

        PARAMETERS
        ----------
        fileName : str
            String input.
        """
        self.__contains_details = False
        inputFile = open(fileName, "r", encoding="utf8")
        lines = inputFile.readlines()
        inputFile.close()
        for line in lines:
            self.__results.append(Performance(float(line)))

    def add(self, performance: Performance):
        """
        The add method takes a Performance as an input and adds it to the results list.

        PARAMETERS
        ----------
        performance : Performance
            Performance input.
        """
        if not isinstance(performance, DetailedClassificationPerformance):
            self.__contains_details = False
        if not isinstance(performance, ClassificationPerformance):
            self.__classification = False
        self.__results.append(performance)

    def numberOfExperiments(self) -> int:
        """
        The numberOfExperiments method returns the size of the results {@link ArrayList}.

        RETURNS
        -------
        int
            The results list.
        """
        return len(self.__results)

    def getErrorRate(self, index: int) -> float:
        """
        The getErrorRate method takes an index as an input and returns the errorRate at given index of results list.

        PARAMETERS
        ----------
        index : int
            Index of results list to retrieve.

        RETURNS
        -------
        float
            The errorRate at given index of results list.
        """
        return self.__results[index].getErrorRate()

    def getAccuracy(self, index: int) -> float:
        """
        The getAccuracy method takes an index as an input. It returns the accuracy of a Performance at given index
        of results list.

        PARAMETERS
        ----------
        index : int
            Index of results list to retrieve.

        RETURNS
        -------
        float
            The accuracy of a Performance at given index of results list.
        """
        return self.__results[index].getAccuracy()

    def meanPerformance(self) -> Performance:
        """
        The meanPerformance method loops through the performances of results list and sums up the errorRates of each
        then returns a new Performance with the mean of that summation.

        RETURNS
        -------
        Performance
            A new Performance with the mean of the summation of errorRates.
        """
        sum_error = 0
        for performance in self.__results:
            sum_error += performance.getErrorRate()
        return Performance(sum_error / len(self.__results))

    def meanClassificationPerformance(self) -> ClassificationPerformance:
        """
        The meanClassificationPerformance method loops through the performances of results list and sums up
        the accuracy of each classification performance, then returns a new classificationPerformance with the mean of
        that summation.

        RETURNS
        -------
        ClassificationPerformance
            A new classificationPerformance with the mean of that summation.
        """
        if len(self.__results) == 0 or not self.__classification:
            return None
        sum_accuracy = 0
        for performance in self.__results:
            sum_accuracy += performance.getAccuracy()
        return ClassificationPerformance(sum_accuracy / len(self.__results))

    def meanDetailedPerformance(self) -> DetailedClassificationPerformance:
        """
        The meanDetailedPerformance method gets the first confusion matrix of results list.
        Then, it adds new confusion matrices as the DetailedClassificationPerformance of other elements of results
        ArrayList' confusion matrices as a DetailedClassificationPerformance.

        RETURNS
        -------
        DetailedCassificationPerformance
            A new DetailedClassificationPerformance with the ConfusionMatrix sum.
        """
        if len(self.__results) == 0 or not self.__contains_details:
            return None
        sum_matrix = self.__results[0].getConfusionMatrix()
        for i in range(1, len(self.__results)):
            sum_matrix.addConfusionMatrix(self.__results[i].getConfusionMatrix())
        return DetailedClassificationPerformance(sum_matrix)

    def standardDeviationPerformance(self) -> Performance:
        """
        The standardDeviationPerformance method loops through the Performances of results list and returns
        a new Performance with the standard deviation.

        RETURNS
        -------
        Performance
            A new Performance with the standard deviation.
        """
        sum_error_rate = 0
        average_performance = self.meanPerformance()
        for performance in self.__results:
            sum_error_rate += math.pow(
                performance.getErrorRate() - average_performance.getErrorRate(), 2
            )
        return Performance(math.sqrt(sum_error_rate / (len(self.__results) - 1)))

    def standardDeviationClassificationPerformance(self) -> ClassificationPerformance:
        """
        The standardDeviationClassificationPerformance method loops through the Performances of results list and
        returns a new ClassificationPerformance with standard deviation.

        RETURNS
        -------
        ClassificationPerformance
            A new ClassificationPerformance with standard deviation.
        """
        if len(self.__results) == 0 or not self.__classification:
            return None
        sum_accuracy = 0
        sum_error_rate = 0
        average_classification_performance = self.meanClassificationPerformance()
        for performance in self.__results:
            sum_accuracy += math.pow(
                performance.getAccuracy()
                - average_classification_performance.getAccuracy(),
                2,
            )
            sum_error_rate += math.pow(
                performance.getErrorRate()
                - average_classification_performance.getErrorRate(),
                2,
            )
        return ClassificationPerformance(
            math.sqrt(sum_accuracy / (len(self.__results) - 1))
        )

    def isBetter(self, experimentPerformance: ExperimentPerformance) -> bool:
        """
        The isBetter method  takes an ExperimentPerformance as an input and returns true if the result of compareTo
        method is positive and false otherwise.

        PARAMETERS
        ----------
        experimentPerformance : ExperimentPerformance
            ExperimentPerformance input.

        RETURNS
        -------
        bool
            True if the experiment performance is better than the given experiment performance.
        """
        return self > experimentPerformance
