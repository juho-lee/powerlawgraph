from defs import *

# sigmoid transform
class Sigmoid(object):
    def __init__(self, c=1.0):
        super(Sigmoid, self).__init__()
        self.c = c

    def __call__(self, x):
        return self.c/(1 + exp(-x))

    def inv(self, y):
        return (log(y) - log(self.c-y))

    def jacobian(self, x):
        return x*(1.-x/self.c)

# softplus transformation
class Softplus(object):
    def __init__(self):
        super(Softplus, self).__init__()

    def __call__(self, x):
        return np.logaddexp(0, x)

    def inv(self, y):
        return y + log(1. - exp(-y))

    def jacobian(self, x):
        return 1 - exp(-x)
