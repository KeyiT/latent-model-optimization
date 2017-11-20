from latent_model_optimization import *
import numpy as np
from numpy import exp



class MingLeiModel(LatentModelOptimizer):

    def latent_model(self, hidden_vars, params):
        e1, e2 = hidden_vars
        theta1, theta2 = params

        beta = 2 * math.pi / 1.55e-9
        L = 100e-6

        M2 = np.matrix([[0.70385 * exp(-1j * 1.073), -0.709911 * exp(-1j * 1.237)],
                        [0.707941 * exp(-1j * 1.134), 0.703113 * exp(-1j * 1.243)]])
        M4 = np.matrix([[0.703036 * exp(-1j * 1.243), -0.70724 * exp(-1j * 1.237)],
                        [0.710595 * exp(-1j * 1.134), 0.703917 * exp(-1j * 1.073)]])

        Ein = np.matrix([e1, e2]).transpose()
        M10 = np.matrix([[exp(-1j * beta * L - 1j * theta1), 0], [0, exp(-1j * beta * L)]])
        M30 = np.matrix([[exp(-1j * beta * L - 1j * theta2), 0], [0, exp(-1j * beta * L)]])

        M50 = M4 * M30 * M2 * M10 * Ein
        E30 = M50.item((0, 0))
        return E30

    def observe(self, params):
        return self.latent_model(
            target_hidden_vars,
            params)

    def model(self, params):
        """
        Override parent model to accommodate complex number.
        :param params:
        :return:
        """
        E30 = self.latent_model(self.hidden_vars, params)
        P30 = np.abs(E30)
        return P30

target_hidden_vars = [0.1, 0.2]
model = MingLeiModel([1.0, 1.2], [1.0, 0.4])
model.optimize(max_iter=30)

print "\nhidden variables:"
print model.hidden_vars
print "target hidden variables:"
print target_hidden_vars
print "model parameters:"
print model.params
model.hidden_vars = target_hidden_vars
results = minimize(model.model, model.params, jac=False)
print results.message
print results.x
print "target minimum:"
print results.fun