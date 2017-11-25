from scipy.optimize import minimize, least_squares
import math
import numpy as np


class LatentModelOptimizer:

    def __init__(self, init_hidden_vars, init_params):
        self.hidden_vars = list(init_hidden_vars)
        self.params = None
        self.set_params(list(init_params))

    def latent_model(self, hidden_vars, params):
        """

        :param hidden_vars: list
        :param params: list
        :return: int, array-like
            Observation.
        """
        raise NotImplementedError

    def set_params(self, params):
        self.params = params

    def model(self, params):
        return self.latent_model(self.hidden_vars, params)

    def model_jac(self, params):
        pass

    def model_hes(self, params):
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
        return self.functional(hidden_vars) - self.observe(self.params)

    def functional_residual_square(self, hidden_vars):
        return self.functional_residual(hidden_vars) ** 2

    def functional_jac(self, hidden_vars):
        raise NotImplementedError

    def functional_hes(self, hidden_vars):
        raise NotImplementedError

    def functional_residual_square_jac(self, hidden_vars):
        return self.functional_jac(hidden_vars) * (self.functional(hidden_vars) - self.observe(self.params)) * 2

    def optimize(self, method=None, jac=False, hes=None, bounds_params=None, bounds_hidden_vars=None, tol=1E-16, max_iter=100):

        for i in range(0, max_iter):

            # estimate the model
            if not jac:
                results = least_squares(self.functional_residual, [0.5, 0.5],
                                        verbose=2, method='dogbox', bounds=bounds_hidden_vars,
                                        ftol=3e-16, xtol=3e-16, gtol=3e-16)
            else:
                results = least_squares(self.functional_residual, [0.5, 0.5], jac=self.functional_jac,
                                        verbose=2, method='dogbox', bounds=bounds_hidden_vars,
                                        ftol=3e-16, xtol=3e-16, gtol=3e-16)

            if not results.success:
                print("model estimation warning: " + results.message)
            self.hidden_vars = results.x

            self.validate_model()

            # minimize the estimated model
            if not jac:
                results = minimize(self.model, self.params, method=method, jac=False,
                                   hess=hes, bounds=bounds_params, tol=1E-16)
            else:
                results = minimize(self.model, self.params, method=method, jac=self.model_jac,
                                   hess=hes, bounds=bounds_params, tol=1E-16)

            if not results.success:
                print("model minimization warning: " + results.message)
            self.set_params(list(results.x))
            print(str(i) + "th iteration: minimum is " + str(results.fun))

            res = self.functional_residual_square(self.hidden_vars)
            print(str(i) + "th iteration: residual is " + str(res))
            if res < tol:
                pass
                # break

            print(self.hidden_vars)
            print(self.params)

            self.validate_model()

    def train(self, sample_parameters, method=None, jac=False, bounds=None, max_iter=1, tol=1E-10):

        def loss_function(hidden_vars):
            loss = []
            for sample in sample_parameters:
                self.set_params(list(sample))
                loss.append(self.functional_residual(hidden_vars))

            return loss

        def loss_function_jac(hidden_vars):
            jac_ = []
            for sample in sample_parameters:
                self.set_params(list(sample))
                tmp = self.functional_jac(hidden_vars)
                jac_.append([tmp[0][0], tmp[0][1]])

            return jac_

        if not jac:
            results = least_squares(loss_function, [0.5, np.pi],
                                    verbose=2, method=method, bounds=bounds, ftol=3e-16, xtol=3e-16, gtol=3e-16)

        else:
            results = least_squares(loss_function, [0.5, np.pi],
                                    jac=loss_function_jac, verbose=2,
                                    method=method, bounds=bounds, ftol=3e-16, xtol=3e-16, gtol=3e-16)

        print(results.message)
        self.hidden_vars = results.x

        self.validate_model()

    def validate_model(self):
        pass