import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

np.random.seed(0)


def generate_data(x_train, t_train, overhaul_intensity, failure_intensity, x_test, num_integration_samples, treatment_strengths):
    # Generate factual outcomes for training set:
    w_o = np.random.uniform(0, 1, size=x_train.shape[1])
    w_ot = np.random.uniform(0, 1, size=x_train.shape[1])
    gamma_o = - expit(w_ot * x_train).sum(axis=1)
    o_train = expit((w_o * x_train).sum(axis=1)
                    + 2 * gamma_o * t_train[:, 0]
                    # + gamma_o * t_train[:, 0] ** 2
                    + np.random.normal(0, 1, size=x_train.shape[0])
                    )
    # Rescale to have same mean
    nu_o = 7
    o_train = o_train * nu_o

    w_f = np.random.uniform(0, 1, size=x_train.shape[1])
    w_ft = np.random.uniform(0, 1, size=x_train.shape[1])
    gamma_f = - expit(w_ft * x_train).sum(axis=1)
    f_train = expit((w_f * x_train).sum(axis=1)
                    + 2 * gamma_f * t_train[:, 0]
                    # + gamma_f * t_train[:, 0] ** 2
                    + np.random.normal(0, 1, size=x_train.shape[0])
                    )
    nu_f = 9
    f_train = f_train * nu_f

    # print('Correlation: ')
    # print('PM - Overhaul: ' + str(np.round(np.corrcoef(t_train[:, 0], o_train)[0, 1], 4)))
    # print('PM - Failures: ' + str(np.round(np.corrcoef(t_train[:, 0], f_train)[0, 1], 4)))

    # Generate all potential outcomes for test set (0 - max treatments):
    # t_test = np.tile(np.cumsum(np.ones(E)) - 1, (x_test.shape[0], 1))
    t_test = np.tile(treatment_strengths, (x_test.shape[0], 1))

    gamma_o_test = - expit(w_ot * x_test).sum(axis=1)
    o_test = expit(np.tile((w_o * x_test).sum(axis=1), (num_integration_samples, 1)).T
                   + 2 * np.tile(gamma_o_test, (num_integration_samples, 1)).T * t_test
                   # + 20 * gamma_o * t_test
                   # + np.tile(gamma_o_test, (num_integration_samples, 1)).T * (t_test ** 2)
                   + np.tile(np.random.normal(0, 1, size=(x_test.shape[0])), (num_integration_samples, 1)).T
                   )
    o_test = o_test * nu_o

    gamma_f_test = - expit(w_ft * x_test).sum(axis=1)
    f_test = expit(np.tile((w_f * x_test).sum(axis=1), (num_integration_samples, 1)).T
                   + 2 * np.tile(gamma_f_test, (num_integration_samples, 1)).T * t_test
                   # + 20 * gamma_f * t_test
                   # + np.tile(gamma_f_test, (num_integration_samples, 1)).T * (t_test ** 2)
                   + np.tile(np.random.normal(0, 1, size=(x_test.shape[0])), (num_integration_samples, 1)).T
                   )
    f_test = f_test * nu_f

    return o_train, f_train, o_test, f_test
