import numpy as np
from abc import ABC, abstractmethod

eps = np.finfo(float).eps


class Operation(ABC):
    def __init__(self, name):
        self.name = name

    @property
    def __name__(self):
        return self.name

    def get_as(self, x ,threshold):
        raise NotImplemented

    def __call__(self, x):
        raise NotImplemented

    def __repr__(self):
        return 'Op ({})'.format(self.name)


class Max(Operation):
    def __init__(self, name='max'):
        super().__init__(name)

    def __call__(self, x):
        return np.max(x, axis=0)

    def get_as(self, x ,threshold):
        # return np.argmax(x).flatten().tolist()
        return np.argwhere(x >= threshold).flatten().tolist()


class Min(Operation):
    def __init__(self, name='min'):
        super().__init__(name)

    def __call__(self, x):
        return np.min(x, axis=0)

    def get_as(self, x ,threshold):
        # return np.argmin(x).flatten().tolist()
        return np.argwhere(x <= threshold).flatten().tolist()


class Mean(Operation):
    def __init__(self, name='mean'):
        super().__init__(name)

    def __call__(self, x):
        return np.mean(x, axis=0)

    def get_as(self, x ,threshold):
        return np.argwhere(x >= threshold).flatten().tolist()


class Sum(Operation):
    def __init__(self, name='sum'):
        super().__init__(name)

    def __call__(self, x):
        return np.sum(x, axis=0)

    def get_as(self, x ,threshold):
        return np.argwhere(x >= threshold / len(x)).flatten().tolist()


class AggregateOperation(Operation):
    def __init__(self, z, name):
        super().__init__(name)
        self.z = z

    def __call__(self, x):
        t = np.power(x, self.z)
        return np.sum(t, 0)

    def get_as(self, x ,threshold):
        return np.argwhere(np.power(x, self.z) >= threshold / len(x)).flatten().tolist()


class MeanOperation(Operation):
    def __init__(self, z, name):
        super().__init__(name)
        self.z = z

    def __call__(self, x):
        t = np.power(x + eps, self.z) / x.shape[0]
        return np.sum(t, 0)

    def get_as(self, x ,threshold):
        x = np.argwhere(np.power(x + eps, self.z) >= threshold).flatten().tolist()
        return x


class SecondMomentMean(MeanOperation):
    def __init__(self, z=2, name='sec_mom_mean'):
        super().__init__(z, name)


class HarmonicMean(MeanOperation):
    def __init__(self, z=-1, name='harm_mean'):
        super().__init__(z, name)


class GeometricMean(MeanOperation):
    def __init__(self, z=0, name='geo_mean'):
        super().__init__(z, name)

