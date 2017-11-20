from scipy.optimize import minimize
import math
import numpy as np


class LatentModelOptimizer:

    def __init__(self, init_hidden_vars, init_params):
        self.hidden_vars = list(init_hidden_vars)
        self.params = list(init_params)

    def latent_model(self, hidden_vars, params):
        """

        :param hidden_vars: list
        :param params: list
        :return: int, array-like
            Observation.
        """
        raise NotImplementedError

    def latent_model_hes(self, hidden_vars, params):
        """

        :param hidden_vars: list
        :param params: list
        :return: int, array-like
            Hessian observation.
        """
        pass

    def model(self, params):
        return self.latent_model(self.hidden_vars, params)

    def model_jac(self, params):
        pass

    def functional(self, hidden_vars):
        return self.latent_model(hidden_vars, self.params)

    def observe(self, params):
        raise NotImplementedError

    def functional_residual(self, hidden_vars):
        """
        Default residual is square of the absolute difference between observation and functional return.
        :param hidden_vars:
        :return:
        """
        return np.power(np.abs(self.functional(hidden_vars) - self.observe(self.params)), 2)

    def functional_jac(self, hidden_vars):
        raise NotImplementedError

    def functional_residual_jac(self, hidden_vars):
        return self.functional_jac(hidden_vars) * (self.functional(hidden_vars) - self.observe(self.params)) * 2

    def optimize(self, method=None, jac=False, bounds_params=None, bounds_hidden_vars=None, tol=1E-16, max_iter=100):

        for i in xrange(0, max_iter):

            # estimate the model
            if not jac:
                results = minimize(self.functional_residual, self.hidden_vars, method=method, jac=False,
                                   bounds=bounds_hidden_vars, tol=1E-16)
            else:
                results = minimize(self.functional_residual, self.hidden_vars, method=method, jac=self.functional_residual_jac,
                                   bounds=bounds_hidden_vars, tol=1E-16)

            if not results.success:
                print "model estimation warning: " + results.message
            self.hidden_vars = results.x

            self.validate_model()

            # minimize the estimated model
            if not jac:
                results = minimize(self.model, self.params, method=method, jac=False, bounds=bounds_params, tol=1E-16)
            else:
                results = minimize(self.model, self.params, method=method, jac=self.model_jac, bounds=bounds_params, tol=1E-16)

            if not results.success:
                print "model minimization warning: " + results.message
            self.params = results.x
            print str(i) + "th iteration: minimum is " + str(results.fun)

            res = self.functional_residual(self.hidden_vars)
            print str(i) + "th iteration: residual is " + str(res)
            if res < tol:
                pass
                # break

            self.validate_model()

    def train(self, sample_parameters, method=None, jac=False, bounds=None, max_iter=1, tol=1E-10):

        def loss_function(hidden_vars):
            loss = 0
            for sample in sample_parameters:
                self.params = sample
                loss += self.functional_residual(hidden_vars)

            return loss

        def loss_function_jac(hidden_vars):
            jac_ = np.ndarray(shape=[1, 2])
            for sample in sample_parameters:
                self.params = sample
                jac_ += self.functional_residual_jac(hidden_vars)

            return jac_

        for i in xrange(0, max_iter):
            if not jac:
                results = minimize(loss_function,
                                   self.hidden_vars, jac=False,
                                   method=method, bounds=bounds, tol=tol)

            else:
                results = minimize(loss_function,
                                   self.hidden_vars, jac=loss_function_jac,
                                   method=method, bounds=bounds, tol=tol)

            print results.message
            self.hidden_vars = results.x
            print str(i) + "th iteration: average functional residual is " + str(results.fun / len(sample_parameters))

        self.validate_model()

    def validate_model(self):
        pass