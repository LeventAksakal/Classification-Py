from Classification.Attribute.ContinuousAttribute import ContinuousAttribute
from Classification.DataSet.DataSet import DataSet
from Classification.Filter.FeatureFilter import FeatureFilter
from Classification.Instance.Instance import Instance


class Normalize(FeatureFilter):

    """
    Constructor for normalize feature filter. It calculates and stores the mean (m) and standard deviation (s) of
    the sample.

    PARAMETERS
    ----------
    dataSet : DataSet
        Instances whose continuous attribute values will be normalized.
    """
    def __init__(self, dataSet: DataSet):
        super().__init__(dataSet)
        self.averageInstance = dataSet.getInstanceList().average()
        self.standardDeviationInstance = dataSet.getInstanceList().standardDeviation()

    """
    Normalizes the continuous attributes of a single instance. For all i, new x_i = (x_i - m_i) / s_i.

    PARAMETERS
    ----------
    instance : Instance
        Instance whose attributes will be normalized.
    """
    def convertInstance(self, instance: Instance):
        for i in range(instance.attributeSize()):
            if isinstance(instance.getAttribute(i), ContinuousAttribute):
                xi = instance.getAttribute(i)
                mi = self.averageInstance.getAttribute(i)
                si = self.standardDeviationInstance.getAttribute(i)
                xi.setValue((xi.getValue() - mi.getValue()) / si.getValue())

    def convertDataDefinition(self):
        pass
