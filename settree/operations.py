import numpy as np
from abc import ABC
eps = np.finfo(float).eps


class Operation(ABC):
    def __init__(self, alpha=1, name=''):
        self.alpha = alpha
        self.name = name

        if self.alpha < 0:
            self.trans = lambda x: x + eps
        else:
            self.trans = lambda x: x

    @property
    def __name__(self):
        return self.name

    def get_as(self, x, threshold):
        raise NotImplemented

    def __call__(self, x):
        raise NotImplemented

    def __repr__(self):
        return 'Op ({})'.format(self.name)


class MeanOperation(Operation):
    def __init__(self, alpha, name):
        super().__init__(alpha, name)

    def __call__(self, x):
        t = np.power(self.trans(x), self.alpha)
        return np.mean(t, 0)

    def get_as(self, x ,threshold):
        x = np.argwhere(np.power(self.trans(x), self.alpha) >= threshold).flatten().tolist()
        return x


class SumOperation(Operation):
    def __init__(self, alpha, name):
        super().__init__(alpha, name)

    def __call__(self, x):
        t = np.power(self.trans(x), self.alpha)
        return np.sum(t, 0)

    def get_as(self, x, threshold):
        x = np.argwhere(np.power(self.trans(x), self.alpha) >= threshold / x.shape[0]).flatten().tolist()
        return x


class Max(Operation):
    def __init__(self, alpha=np.inf, name='max'):
        super().__init__(alpha, name)

    def __call__(self, x):
        return np.max(x, axis=0)

    def get_as(self, x, threshold):
        return np.argwhere(x >= threshold).flatten().tolist()


class Min(Operation):
    def __init__(self, alpha=-np.inf, name='min'):
        super().__init__(alpha, name)

    def __call__(self, x):
        return np.min(x, axis=0)

    def get_as(self, x, threshold):
        return np.argwhere(x <= threshold).flatten().tolist()


class Sum(SumOperation):
    def __init__(self, alpha=1, name='sum'):
        super().__init__(alpha, name)


class SecondMomentSum(SumOperation):
    def __init__(self, alpha=2, name='sec_mom_sum'):
        super().__init__(alpha, name)


class Mean(Operation):
    def __init__(self, alpha=1, name='mean'):
        super().__init__(alpha, name)

    def __call__(self, x):
        return np.mean(x, axis=0)

    def get_as(self, x ,threshold):
        return np.argwhere(x >= threshold).flatten().tolist()


class SecondMomentMean(MeanOperation):
    def __init__(self, alpha=2, name='sec_mom_mean'):
        super().__init__(alpha, name)


class HarmonicMean(MeanOperation):
    def __init__(self, alpha=-1, name='harm_mean'):
        super().__init__(alpha, name)


class GeometricMean(MeanOperation):
    def __init__(self, alpha=0, name='geo_mean'):
        super().__init__(alpha, name)

