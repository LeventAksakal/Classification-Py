from abc import abstractmethod


class Clustering(object):
    """Class for representing clustering classfiers

    Args:
        object (_type_): _description_
    """
    @abstractmethod
    def train(self):
        pass
