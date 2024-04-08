from typing import List
from Classification.StatisticalTest.MultivariateTest import MultivariateTest
from Classification.Performance.ExperimentPerformance import ExperimentPerformance


class KruskalWallis(MultivariateTest):
    """
    Class representing the KruskalWallis statistical test.

    Attributes:
        __H (float): The H-value computed by the KruskalWallis test.
    """

    __H: float

    def compare(self, experiment_performances: List[ExperimentPerformance]):
        """
        This method calculates an H score with KruskalWallis statistical test by comparing the performance results
        of different classifiers on various datasets.

        Args: List of experiment performances

        Returns: Calculated H score
        """
        group_ranks = []
        for experiment_performance in experiment_performances:
            group_ranks.append(
                sum(
                    result.getAccuracy()
                    for result in experiment_performance.getResults()
                )
            )

        H = 12 / (len(group_ranks) * (len(group_ranks) + 1)) * sum(
            (r**2) / len(experiment_performance.getResults())
            for r, experiment_performance in zip(group_ranks, experiment_performances)
        ) - 3 * (len(group_ranks) + 1)

        self.__H = H
        return self.__H

    def get_HValue(self):
        return self.__H
