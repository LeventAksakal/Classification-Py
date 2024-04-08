from typing import List
from Classification.StatisticalTest.MultivariateTest import MultivariateTest
from Classification.Performance.ExperimentPerformance import ExperimentPerformance


class Anova(MultivariateTest):
    """
    Class representing the Anova statistical test.

    Attributes:
        __f (float): The F-value computed by the Anova test.
    """

    __f: float

    def compare(self, experiment_performances: List[ExperimentPerformance]):
        """
        This method calculates an F score with ANOVA by comparing the performance results
        of different classifiers on various datasets.

        Args: List of experiment performances

        Returns: Calculated F score
        """
        total_samples = sum(
            experiment.numberOfExperiments() for experiment in experiment_performances
        )
        total_groups = len(experiment_performances)

        grand_mean = (
            sum(
                experiment.meanPerformance().getAccuracy()
                * experiment.numberOfExperiments()
                for experiment in experiment_performances
            )
            / total_samples
        )

        ss_between = sum(
            experiment.numberOfExperiments()
            * ((experiment.meanPerformance().getAccuracy() - grand_mean) ** 2)
            for experiment in experiment_performances
        )

        ss_within = sum(
            sum(
                (performance.getAccuracy() - experiment.meanPerformance().getAccuracy())
                ** 2
                for performance in experiment.__results
            )
            for experiment in experiment_performances
        )

        df_between = total_groups - 1
        df_within = total_samples - total_groups

        ms_between = ss_between / df_between
        ms_within = ss_within / df_within

        self.__f = ms_between / ms_within
        return self.__f

    def get_FValue(self):
        return self.__f
