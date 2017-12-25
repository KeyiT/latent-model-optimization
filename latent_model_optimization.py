from scipy.optimize import minimize, least_squares
import math
import numpy as np


class LatentModelOptimizer(object):

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

    def functional_residual(self, hidden_vars, observe_=None):
        """
        Default residual is square of the absolute difference between observation and functional return.
        :param hidden_vars:
        :param observe_:
        :return:
        """
        if observe_ is None:
            return self.functional(hidden_vars) - self.observe(self.params)
        else:
            return self.functional(hidden_vars) - observe_

    def functional_residual_square(self, hidden_vars):
        return self.functional_residual(hidden_vars) ** 2

    def functional_jac(self, hidden_vars):
        raise NotImplementedError

    def functional_hes(self, hidden_vars):
        raise NotImplementedError

    def functional_residual_square_jac(self, hidden_vars):
        return self.functional_jac(hidden_vars) * (self.functional(hidden_vars) - self.observe(self.params)) * 2

    def train_optimize_with_smart_sampling(self, initia_params,
                                           train_method='trf', optimize_method='Newton-CG',
                                           train_jac=False, optimize_jac=False,
                                           bounds_hidden_vars=(-np.inf, np.inf), bounds_params=None,
                                           train_tol=1E-16, optimize_tol=1E-16,
                                           max_iter=10):

        observes_ = list()
        sample_parameters = list()
        sample_parameters.append(initia_params)

        for i in range(0, max_iter):

            observes_.append(self.observe(sample_parameters[-1]))

            tmp_hidden_vars = list()
            tmp_hidden_vars.append(self.hidden_vars)
            self.train(sample_parameters=sample_parameters, init_guess_=tmp_hidden_vars, observes=observes_,
                       method=train_method, jac=train_jac, bounds=bounds_hidden_vars, tol=train_tol)

            self.validate_model()

            self.optimize(init_guess_=self.params, method=optimize_method,
                          jac=optimize_jac,
                          bounds=bounds_params, tol=optimize_tol)
            sample_parameters.append(self.params)

            self.validate_model()

            print(str(i) + "th iteration: minimum is " + str(self.model(self.params)))

            res = self.functional_residual_square(self.hidden_vars)
            print(str(i) + "th iteration: residual is " + str(res))

            print(self.hidden_vars)
            print(self.params)

    def train_optimize_with_batch_sampling(self, initial_hidden_vars, sample_params,
                                           train_method='trf', optimize_method=None,
                                           train_jac=False, optimize_jac=False,
                                           bounds_hidden_vars=(-np.inf, np.inf), bounds_params=None,
                                           train_tol=1E-16, optimize_tol=1E-16):

        self.train(sample_parameters=sample_params, init_guess_=initial_hidden_vars, observes=None,
                   method=train_method, jac=train_jac, bounds=bounds_hidden_vars, tol=train_tol)

        self.validate_model()

        self.optimize(init_guess_=self.params, method=optimize_method,
                      jac=optimize_jac,
                      bounds=bounds_params, tol=optimize_tol)

        self.validate_model()

    def train(self, sample_parameters, init_guess_, observes=None, method=None, jac=False, bounds=None, tol=1E-10):

        if observes is None:
            observes_ = list(map(
                lambda sample: self.observe(list(sample)),
                sample_parameters
            ))
        else:
            observes_ = observes

        def loss_function(hidden_vars):
            loss = []
            for i in range(0, len(sample_parameters)):
                sample = sample_parameters[i]
                self.params = list(sample)
                loss.append(self.functional_residual(hidden_vars, observe_=observes_[i]))

            return loss

        def loss_function_jac(hidden_vars):
            jac_ = []
            for sample in sample_parameters:
                self.params = list(sample)
                tmp = self.functional_jac(hidden_vars)
                jac_.append([tmp[0][0], tmp[0][1]])

            return jac_

        if not jac:
            for ini in init_guess_:
                results = least_squares(loss_function, ini,
                                    verbose=1, method=method, bounds=bounds, ftol=3e-16, xtol=3e-16, gtol=3e-16)
                if results.success and np.sum(np.array(results.fun)) < tol:
                    self.hidden_vars = results.x
                    break

        else:
            for ini in init_guess_:
                print(ini)
                results = least_squares(loss_function, ini,
                                        jac=loss_function_jac, verbose=1,
                                        method=method, bounds=bounds, ftol=3e-16, xtol=3e-16, gtol=3e-16)
                if results.success and np.sum(np.array(results.fun)) < tol:
                    self.hidden_vars = results.x
                    break

    def optimize(self, init_guess_, method=None, jac=False, bounds=None, tol=1E-16):
        # minimize the estimated model
        if not jac:
            results = minimize(self.model, init_guess_, method=method,
                               jac=False,
                               bounds=bounds, tol=tol)
        else:
            results = minimize(self.model, init_guess_, method=method,
                               jac=self.model_jac, hess=self.model_hes,
                               bounds=bounds, tol=tol)

        print(results.message())
        self.set_params(results.x)

    def validate_model(self):
        pass