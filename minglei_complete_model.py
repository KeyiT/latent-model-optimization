from latent_model_optimization import *
import numpy as np
from numpy import exp
from scipy.optimize import minimize
import math
import sympy as sym
from sympy.utilities.lambdify import lambdify


class MingLeiModel(LatentModelOptimizer):

    def __init__(self, init_hidden_vars, init_params):
        self.hidden_vars = list(init_hidden_vars)
        self.params = list(init_params)

        # generate lambda function of jacobin of model with respect to hidden variables
        h1, h2 = sym.symbols('h1, h2', real=True)
        theta1, theta2 = sym.symbols('theta1, theta2', real=True)
        jac_hidden_sym_ = MingLeiModel._latent_model_jac_hidden_sym(h1, h2, theta1, theta2)
        self._jac_hidden = lambdify(
            (h1, h2, theta1, theta2),
            jac_hidden_sym_, modules='numpy')

        # generate lambda function of jacobin of model with respect to parameters
        h1, h2 = sym.symbols('h1, h2', real=True)
        theta1, theta2 = sym.symbols('theta1, theta2', real=True)
        jac_params_sym_ = MingLeiModel._latent_model_jac_params_sym(h1, h2, theta1, theta2)
        self._jac_params = lambdify(
            (h1, h2, theta1, theta2),
            jac_params_sym_, modules='numpy')

        # test lambda function of jacobins
        print self._jac_params(0.1, 0.1, 0.1, 0.1)
        print self._jac_hidden(0.1, 0.1, 0.1, 0.1)

        # generate lambda function of latent model
        h1, h2 = sym.symbols('h1, h2', real=True)
        theta1, theta2 = sym.symbols('theta1, theta2', real=True)
        self._latent_model = lambdify(
            (h1, h2, theta1, theta2),
            sym.re(MingLeiModel._latent_model_sym(h1, h2, theta1, theta2)), modules='numpy')


    @classmethod
    def _latent_model_sym(cls, h1, h2, theta1, theta2):
        """
        Symbolic latent model
        :param h1:
        :param h2:
        :param theta1:
        :param theta2:
        :return: symbolic model
        """
        beta = 2 * math.pi / 1.55e-9
        L = 100e-6
        k = math.sqrt(0.5)
        t = math.sqrt(0.5)

        M2 = sym.Matrix([[t, -k],
                         [k, t]])
        M4 = sym.Matrix([[t, -k],
                         [k, t]])

        e1 = sym.sqrt(h1) * sym.exp(-1j * h2)
        e2 = sym.sqrt(1 - h1)
        Ein = sym.Matrix([[e1], [e2]])
        M10 = sym.Matrix([[sym.exp(-1j * beta * L - 1j * theta1), 0], [0, sym.exp(-1j * beta * L)]])
        M30 = sym.Matrix([[sym.exp(-1j * beta * L - 1j * theta2), 0], [0, sym.exp(-1j * beta * L)]])

        M50 = M4 * M30 * M2 * M10 * Ein
        r30 = M50[0, 0]

        p3 = sym.conjugate(r30) * r30

        return p3

    @classmethod
    def _latent_model_jac_params_sym(cls, h1, h2, theta1, theta2):
        """
        Symbolic model Jacobin with respect to model parameters
        :param h1:
        :param h2:
        :param theta1:
        :param theta2:
        :return:
        """

        p3 = MingLeiModel._latent_model_sym(h1, h2, theta1, theta2)

        dydh1 = sym.re(sym.diff(p3, theta1))
        dydh2 = sym.re(sym.diff(p3, theta2))

        return sym.Matrix([dydh1, dydh2]).transpose()

    @classmethod
    def _latent_model_jac_hidden_sym(cls, h1, h2, theta1, theta2):
        """
        Symbolic model jacobin with respect to hidden variables
        :param h1:
        :param h2:
        :param theta1:
        :param theta2:
        :return:
        """

        p3 = MingLeiModel._latent_model_sym(h1, h2, theta1, theta2)

        dydh1 = sym.re(sym.diff(p3, h1))
        dydh2 = sym.re(sym.diff(p3, h2))

        return sym.Matrix([dydh1, dydh2]).transpose()

    def latent_model(self, hidden_vars, params):
        # Minglei upadte: h1: amplitude of e1 vector; h2: relative phase between e1 and e2
        h1, h2 = hidden_vars
        theta1, theta2 = params

        return self._latent_model(h1, h2, theta1, theta2)

    def model_jac(self, params):
        h1, h2 = self.hidden_vars
        theta1, theta2 = params

        h1 = min(h1, 1.0)
        return self._jac_params(h1, h2, theta1, theta2)

    def functional_jac(self, hidden_vars):
        h1, h2 = hidden_vars
        theta1, theta2 = self.params

        h1 = min(h1, 1.0)
        return self._jac_hidden(h1, h2, theta1, theta2)

    def observe(self, params):
        return self.latent_model(
            target_hidden_vars,
            params)

    def validate_model(self):
        if 0 <= self.params[0] <= 2 * math.pi and \
                                0 <= self.params[1] <= 2 * math.pi and \
                                0 <= self.hidden_vars[0] <= 1:
            return True
        else:
            raise ValueError


target_hidden_vars = [0.1, 0.4*math.pi]
model = MingLeiModel([0.5, 0.5], [0.5, 0.5])

# train model
sample_params1 = np.linspace(0, math.pi*2, 20)
sample_params2 = np.linspace(0, math.pi*2, 20)
sample_params = []
for sample1 in sample_params1:
    for sample2 in sample_params2:
        sample_params.append([sample1, sample2])

model.train(sample_parameters=sample_params, bounds=((0, 1), (0, 2 * math.pi)), method='TNC', jac=True)
print "\nhidden variables:"
print model.hidden_vars
print "target hidden variables:"
print target_hidden_vars

# find minimum
results = minimize(model.model,
                   np.ndarray(shape=[2], buffer=np.array([0.1, 0.1])),
                   jac=False, tol=1E-16,
                   bounds=((0, 2 * math.pi), (0, 2 * math.pi)))
print results.message
print "model parameters:"
print model.params
print "minimum:"
print results.fun

model.hidden_vars = target_hidden_vars
results = minimize(model.model,
                   np.ndarray(shape=[2], buffer=np.array([0.1, 0.1])),
                   method='TNC', jac=model.model_jac, tol=1E-16,
                   bounds=((0, 2 * math.pi), (0, 2 * math.pi)))
print results.message
print "target model parameters:"
print results.x
print "target minimum:"
print results.fun

"""
model.hidden_vars = np.ndarray(shape=[2], buffer=np.array([0.6, 0.08193647]))
results = minimize(model.model,
                   np.ndarray(shape=[2], buffer=np.array([0.05092699, 0.0])),
                   method='TNC', tol=1E-16, jac=model.model_jac,
                   bounds=((0, 2 * math.pi), (0, 2 * math.pi)))
print results.message
print "model parameters:"
print "target model parameters:"
print results.x
print "target minimum:"
print results.fun
print "\n"

model.hidden_vars = np.ndarray(shape=[2], buffer=np.array([0.1, 0.1]))
model.params = np.ndarray(shape=[2], buffer=np.array([0.1, 0.1]))
model.optimize(max_iter=100,
               jac=True,
               method='TNC',
               bounds_hidden_vars=((0, 1), (None, None)),
               bounds_params=((0, 2*math.pi), (0, 2*math.pi)))

print "\nhidden variables:"
print model.hidden_vars
print "target hidden variables:"
print target_hidden_vars


# results = minimize(model.model, model.params, jac=model.model_jac, bounds=((0, 2*math.pi), (0, 2*math.pi)))
#print results.message
print "model parameters:"
print model.params
print "minimum:"
print model.latent_model(model.hidden_vars, model.params)

model.hidden_vars = target_hidden_vars
results = minimize(model.model,
                   np.ndarray(shape=[2], buffer=np.array([0.7, 0.2])),
                   method='TNC', tol=1E-21, jac=model.model_jac,
                   bounds=((0, 2 * math.pi), (0, 2 * math.pi)))
print results.message
print "model parameters:"
print "target model parameters:"
print results.x
print "target minimum:"
print results.fun
"""

