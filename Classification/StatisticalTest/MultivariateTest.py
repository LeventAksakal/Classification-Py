from abc import abstractmethod


class MultivariateTest(object):
    """Class for representing multivariate test

    Args:
        object (_type_): _description_
    """
    @abstractmethod
    def compare(self):
        pass
