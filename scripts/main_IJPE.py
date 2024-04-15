import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import click
import os

from scipy.integrate import romb
from scipy.special import expit

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from utils.preprocess_data import preprocess_mm, preprocess_mm2
from src.utils.evaluation import mise
from src.data.generate_data import generate_data

from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

from SCIGAN.SCIGAN import SCIGAN_Model
from src.utils.evaluation_utils import compute_eval_metrics, get_model_predictions
from nn_supervised import SCIGAN_Supervised

plt.rcParams['figure.dpi'] = 300
plt.rc('font', size=14)
plt.style.use('science')
plt.rcParams['font.family'] = 'sans-serif'

linestyles = ['-', '--', '-.', (0, (5, 5)), ':']

np.random.seed(0)
tf.compat.v1.set_random_seed(0)

DIR = 'C:/Users/' #Set path
if not(os.path.isdir(DIR)):
    DIR = 'C:/Users/' #Set path

@click.command()
@click.option('--dir', type=str, default=DIR)
@click.option('--bias', type=bool, default=False)
@click.option('--lambd', type=float, default=5)
def main(dir, bias=False, lambd=0):
    # GET DATA
    # Load data (not distributed with this repo)

    # Set max t:
    max_t = 20
    t[t > max_t] = max_t
    t = t / max_t

    # Results dictionaries
    results_supervised = dict()
    results_supervised['MISE_o'] = []
    results_supervised['MISE_f'] = []
    results_supervised['PE'] = []

    results_supervised['PCR'] = []

    results_individual = dict()
    results_individual['MISE_o'] = []
    results_individual['MISE_f'] = []
    results_individual['PE'] = []
    results_individual['PCR'] = []

    results_average = dict()
    results_average['PE'] = []
    results_average['PCR'] = []

    # HYPERPARAMETERS TO OPTIMIZE:
    batch_size_range = [32, 64]                        # [64, 128, 256]
    h_dim_range = [32, 64]   # [4, 8, 16]              # [32, 64, 128]
    h_inv_eqv_dim_range = [16, 32]   # [8, 16, 32]     # [16, 32, 64, 128]
    num_dosage_samples_range = [2]  # [1, 2, 3]        # 5

    n_iter = 5

    # Print settings:
    print('Batch size: \t' + str(batch_size_range))
    print('Hidden dim: \t' + str(h_dim_range))
    print('H Inv Eq D: \t' + str(h_inv_eqv_dim_range))
    print('Num Dosage: \t' + str(num_dosage_samples_range))
    print('Iterations: \t' + str(n_iter))
    print('Bias: \t\t' + str(bias))
    if bias:
        print('Lambda: \t\t' + str(lambd))

    for iter in range(n_iter):
        print('\nITERATION ' + str(iter + 1))

        # Split data:
        x_train, x_test, t_train, _ = train_test_split(x, t, test_size=0.2, random_state=0)

        # Add selection bias
        if bias:
            # Random assignment:
            # t_train_uniform = np.random.uniform(0, 1, len(x_train))  # * max_t
            # Selection bias:
            factor = expit(np.sum(np.random.uniform(0, 1, size=x_train.shape[1]) * x_train, axis=1))
            t_train = np.random.beta(a=1 + lambd * factor / 10, b=1 + lambd * factor)
            # Add extra dimension:
            t_train = t_train[:, np.newaxis]

        # Step size "continuous" outcome:
        samples_power_of_two = 11
        num_integration_samples = 2 ** samples_power_of_two + 1
        step_size = 1. / num_integration_samples * max_t
        # Linear
        treatment_strengths = np.linspace(np.finfo(float).eps, 1, num_integration_samples) #* t_train.max()

        # Generate semi-synthetic data:
        o_train, f_train, o_test, f_test = generate_data(x_train, t_train, overhaul_intensity, failure_intensity,
                                                         x_test, num_integration_samples, treatment_strengths)

        x_train, x_val, t_train, t_val, o_train, o_val, f_train, f_val = train_test_split(x_train, t_train, o_train,
                                                                                          f_train, test_size=0.25,
                                                                                          random_state=0)

        max_pm = -1
        # Costs:
        # Show average:
        # plt.figure(figsize=(6, 3))
        # plt.title('Actual cost in terms of PM')
        # plt.plot(treatment_strengths[0:max_pm] * max_t, (treatment_strengths * cost_pm)[0:max_pm] * max_t,
        #          linestyle=':', label='PM')
        # plt.plot(treatment_strengths[0:max_pm] * max_t, (o_test.mean(axis=0) * cost_o)[0:max_pm], linestyle='--',
        #          label='Overhauls')
        # plt.plot(treatment_strengths[0:max_pm] * max_t, (f_test.mean(axis=0) * cost_f)[0:max_pm], linestyle='-.',
        #          label='Failures')
        # plt.plot(treatment_strengths[0:max_pm] * max_t,
        #          (treatment_strengths * cost_pm)[0:max_pm] * max_t + (o_test.mean(axis=0) * cost_o)[0:max_pm] + (
        #                                                                                                                     f_test.mean(
        #                                                                                                                         axis=0) * cost_f)[
        #                                                                                                         0:max_pm],
        #          label='Total')
        # best_treat = np.argmin(
        #     (treatment_strengths * cost_pm)[0:max_pm] * max_t + (o_test.mean(axis=0) * cost_o)[0:max_pm] + (f_test.mean(
        #         axis=0) * cost_f)[0:max_pm])
        # # plt.gca().set_ylim(bottom=None, top=treatment_strengths[max_pm] * cost_pm)
        # plt.vlines(x=treatment_strengths[best_treat] * max_t,
        #            ymin=0,
        #            ymax=(treatment_strengths * cost_pm)[best_treat] * max_t + (o_test.mean(axis=0) * cost_o)[
        #                best_treat] + (f_test.mean(axis=0) * cost_f)[best_treat],
        #            linestyle='solid',
        #            color='k'
        #            )
        # plt.plot(treatment_strengths[best_treat] * max_t,
        #          (treatment_strengths * cost_pm)[best_treat] * max_t + (o_test.mean(axis=0) * cost_o)[best_treat] +
        #          (f_test.mean(axis=0) * cost_f)[best_treat],
        #          marker='x',
        #          color='k'
        #          )
        # plt.ylabel('Cost')
        # plt.xlabel('PM')
        # plt.legend()
        # plt.show()

        # SCIGAN Parameters - default:
        params = {'num_features': x_train.shape[1],
                  'num_treatments': 1,  # Not used
                  'num_dosage_samples': 2,  # Sensitive! (1-2-3-4-5)
                  'export_dir': '',
                  'alpha': 1.0,  # Lambda in paper (strength of supervised loss) - default = 1.0
                  'batch_size': 256,
                  'h_dim': 32,  # 32
                  'h_inv_eqv_dim': 16  # 16 - 32 good values
                  }

        # Get/create working dir:
        if bias:
            wkdir = dir + 'Bias_' + str(bias) + '_Lambda_' + str(lambd)
        else:
            wkdir = dir + 'Bias_' + str(bias)
        if not os.path.isdir(wkdir):
            os.mkdir(wkdir)
        if not os.path.isdir(wkdir + '/Best_Supervised'):
            os.mkdir(wkdir + '/Best_Supervised')
        if not os.path.isdir(wkdir + '/Best_SCIGAN'):
            os.mkdir(wkdir + '/Best_SCIGAN')
        os.chdir(wkdir)

        print('Overhauls: ')
        # Supervised:
        model_o = SCIGAN_Supervised(params)
        model_o.tune(x_train, t_train, o_train, x_val, t_val, o_val, batch_size_range, h_dim_range)

        # params_best = params
        # params_best['batch_size'] = model_o.batch_size
        # params_best['h_dim'] = model_o.h_dim

        # model_o = SCIGAN_Supervised(params_best)
        # model_o.train(Train_X=x_train, Train_T=np.zeros(len(x_train)).astype(int), Train_D=t_train, Train_Y=o_train,
        #               verbose=False)

        o_pred_naive = []
        mises_o_naive = []

        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, ["serve"], os.getcwd() + '/Best_Supervised')

            for patient, outcomes in zip(x_test, o_test):
                for treatment_idx in range(params['num_treatments']):
                    test_data = dict()
                    test_data['x'] = np.repeat(np.expand_dims(patient, axis=0), num_integration_samples, axis=0)
                    test_data['t'] = np.repeat(treatment_idx, num_integration_samples)  # Always 1 in our case
                    test_data['d'] = treatment_strengths

                    pred_dose_response = get_model_predictions(sess=sess,
                                                               num_treatments=1,
                                                               num_dosage_samples=model_o.num_dosage_samples,
                                                               test_data=test_data)
                    o_pred_naive.append(pred_dose_response)

                    mise = romb(np.square(outcomes - pred_dose_response), dx=step_size)
                    # mise = romb(np.square(o_test - pred_dose_response), dx=step_size)
                    mises_o_naive.append(mise)

        mise_total_o_naive = np.mean(mises_o_naive)
        print('MISE = (Overhauls) - Naive \t' + str(np.round(mise_total_o_naive, 4)))
        print('======================================================================')

        # plt.figure(figsize=(4, 2))
        # plt.title('Overhauls - Naive')
        # plt.plot(treatment_strengths, np.mean(o_test, axis=0), linestyle='none', marker='.')
        # plt.plot(treatment_strengths, np.mean(o_pred_naive, axis=0), linestyle='none', marker='+')
        # plt.legend(['Actual', 'Predicted'])
        # plt.show()

        tf.compat.v1.reset_default_graph()

        # Causal
        model_o = SCIGAN_Model(params)
        model_o.tune(x_train, t_train, o_train, x_val, t_val, o_val, batch_size_range, h_dim_range, h_inv_eqv_dim_range,
                     num_dosage_samples_range)

        # params_best = params
        # params_best['batch_size'] = model_o.batch_size
        # params_best['h_dim'] = model_o.h_dim
        # params_best['h_inv_eqv_dim'] = model_o.h_inv_eqv_dim
        # params_best['num_dosage_samples'] = model_o.num_dosage_samples
        #
        # model_o = SCIGAN_Model(params_best)
        # model_o.train(Train_X=x_train, Train_T=np.zeros(len(x_train)).astype(int), Train_D=t_train, Train_Y=o_train,
        #               verbose=False)

        # Predict potential outcomes and compute MISE:
        mises_o = []
        o_pred = []

        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, ["serve"], os.getcwd() + '/Best_SCIGAN')

            for patient, outcomes in zip(x_test, o_test):
                for treatment_idx in range(params['num_treatments']):
                    test_data = dict()
                    test_data['x'] = np.repeat(np.expand_dims(patient, axis=0), num_integration_samples, axis=0)
                    test_data['t'] = np.repeat(treatment_idx, num_integration_samples)  # Always 1 in our case
                    test_data['d'] = treatment_strengths

                    pred_dose_response = get_model_predictions(sess=sess,
                                                               num_treatments=1,
                                                               num_dosage_samples=model_o.num_dosage_samples,
                                                               test_data=test_data)
                    o_pred.append(pred_dose_response)

                    mise = romb(np.square(outcomes - pred_dose_response), dx=step_size)
                    mises_o.append(mise)

        mise_total_o = np.mean(mises_o)
        print('MISE = (Overhauls) \t' + str(np.round(mise_total_o, 4)))
        print('======================================================================')

        tf.compat.v1.reset_default_graph()

        # plt.figure(figsize=(4, 2))
        # plt.title('Overhauls')
        # plt.plot(treatment_strengths, np.mean(o_test, axis=0), linestyle='none', marker='.')
        # plt.plot(treatment_strengths, np.mean(o_pred, axis=0), linestyle='none', marker='+')
        # plt.legend(['Actual', 'Predicted'])
        # plt.show()

        print('Failures: ')
        model_f = SCIGAN_Supervised(params)
        model_f.tune(x_train, t_train, f_train, x_val, t_val, f_val, batch_size_range, h_dim_range)

        # params_best = params
        # params_best['batch_size'] = model_o.batch_size
        # params_best['h_dim'] = model_o.h_dim
        
        # model_f = SCIGAN_Supervised(params_best)
        # model_f.train(Train_X=x_train, Train_T=np.zeros(len(x_train)).astype(int), Train_D=t_train, Train_Y=f_train,
        #               verbose=False)

        # Predict potential outcomes and compute MISE:
        mises_f_naive = []
        f_pred_naive = []

        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, ["serve"], os.getcwd() + '/Best_Supervised')

            for patient, outcomes in zip(x_test, f_test):
                for treatment_idx in range(params['num_treatments']):
                    test_data = dict()
                    test_data['x'] = np.repeat(np.expand_dims(patient, axis=0), num_integration_samples, axis=0)
                    test_data['t'] = np.repeat(treatment_idx, num_integration_samples)  # Always 1 in our case
                    test_data['d'] = treatment_strengths

                    pred_dose_response = get_model_predictions(sess=sess,
                                                               num_treatments=1,
                                                               num_dosage_samples=model_f.num_dosage_samples,
                                                               test_data=test_data)

                    f_pred_naive.append(pred_dose_response)

                    mise = romb(np.square(outcomes - pred_dose_response), dx=step_size)
                    mises_f_naive.append(mise)

        mise_total_f_naive = np.mean(mises_f_naive)
        print('MISE (Failures) - Naive \t= ' + str(np.round(mise_total_f_naive, 4)))
        print('======================================================================')

        tf.compat.v1.reset_default_graph()

        # plt.figure(figsize=(4, 2))
        # plt.title('Failures - Naive')
        # plt.plot(treatment_strengths, np.mean(f_test, axis=0), linestyle='none', marker='.')
        # plt.plot(treatment_strengths, np.mean(f_pred_naive, axis=0), linestyle='none', marker='+')
        # plt.legend(['Actual', 'Predicted'])
        # plt.show()

        model_f = SCIGAN_Model(params)
        model_f.tune(x_train, t_train, f_train, x_val, t_val, f_val, batch_size_range, h_dim_range, h_inv_eqv_dim_range,
                     num_dosage_samples_range)

        # params_best = params
        # params_best['batch_size'] = model_o.batch_size
        # params_best['h_dim'] = model_o.h_dim
        # params_best['h_inv_eqv_dim'] = model_o.h_inv_eqv_dim
        # params_best['num_dosage_samples'] = model_o.num_dosage_samples
        #
        # model_f = SCIGAN_Model(params_best)
        # model_f.train(Train_X=x_train, Train_T=np.zeros(len(x_train)).astype(int), Train_D=t_train, Train_Y=f_train,
        #               verbose=False)

        # Predict potential outcomes and compute MISE:
        mises_f = []
        f_pred = []

        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, ["serve"], os.getcwd() + '/Best_SCIGAN')

            for patient, outcomes in zip(x_test, f_test):
                for treatment_idx in range(params['num_treatments']):
                    test_data = dict()
                    test_data['x'] = np.repeat(np.expand_dims(patient, axis=0), num_integration_samples, axis=0)
                    test_data['t'] = np.repeat(treatment_idx, num_integration_samples)  # Always 1 in our case
                    test_data['d'] = treatment_strengths

                    pred_dose_response = get_model_predictions(sess=sess,
                                                               num_treatments=1,
                                                               num_dosage_samples=model_f.num_dosage_samples,
                                                               test_data=test_data)
                    f_pred.append(pred_dose_response)

                    mise = romb(np.square(outcomes - pred_dose_response), dx=step_size)
                    mises_f.append(mise)

        mise_total_f = np.mean(mises_f)
        print('MISE (Failures) \t= ' + str(np.round(mise_total_f, 4)))
        print('======================================================================')

        tf.compat.v1.reset_default_graph()

        # plt.figure(figsize=(4, 2))
        # plt.title('Failures')
        # plt.plot(treatment_strengths, np.mean(f_test, axis=0), linestyle='none', marker='.')
        # plt.plot(treatment_strengths, np.mean(f_pred, axis=0), linestyle='none', marker='+')
        # plt.legend(['Actual', 'Predicted'])
        # plt.show()

        # Plot MISE distribution
        # plt.boxplot([mises_o_naive, mises_o])
        # plt.show()
        # plt.boxplot([mises_f_naive, mises_f])
        # plt.show()

        # Optimize costs:
        o_pred = np.array(o_pred)
        o_pred_naive = np.array(o_pred_naive)
        f_pred = np.array(f_pred)
        f_pred_naive = np.array(f_pred_naive)

        # Estimated costs -- on average:
        # plt.figure(figsize=(6, 3))
        # plt.title('Estimated cost in terms of PM NAIVE')
        # plt.plot(treatment_strengths[0:max_pm] * max_t, (treatment_strengths*cost_pm)[0:max_pm] * max_t, linestyle=':', label='PM')
        # plt.plot(treatment_strengths[0:max_pm] * max_t, (o_pred_naive.mean(axis=0)*cost_o)[0:max_pm], linestyle='--', label='Overhauls')
        # plt.plot(treatment_strengths[0:max_pm] * max_t, (f_pred_naive.mean(axis=0)*cost_f)[0:max_pm], linestyle='-.', label='Failures')
        # plt.plot(treatment_strengths[0:max_pm] * max_t,
        #          (treatment_strengths * cost_pm)[0:max_pm] * max_t + (o_pred_naive.mean(axis=0) * cost_o)[0:max_pm] + (f_pred_naive.mean(
        #              axis=0) * cost_f)[0:max_pm], label='Total')
        # # plt.gca().set_ylim(bottom=None, top=treatment_strengths[max_pm]*cost_pm)
        # plt.ylabel('Cost')
        # plt.xlabel('PM')
        # plt.legend()
        # plt.show()

        # plt.figure(figsize=(6, 3))
        # plt.title('Estimated cost in terms of PM')
        # plt.plot(treatment_strengths[0:max_pm] * max_t, (treatment_strengths*cost_pm)[0:max_pm] * max_t, linestyle=':', label='PM')
        # plt.plot(treatment_strengths[0:max_pm] * max_t, (o_pred.mean(axis=0)*cost_o)[0:max_pm], linestyle='--', label='Overhauls')
        # plt.plot(treatment_strengths[0:max_pm] * max_t, (f_pred.mean(axis=0)*cost_f)[0:max_pm], linestyle='-.', label='Failures')
        # plt.plot(treatment_strengths[0:max_pm] * max_t,
        #          (treatment_strengths * cost_pm)[0:max_pm] * max_t + (o_pred.mean(axis=0) * cost_o)[0:max_pm] + (f_pred.mean(
        #              axis=0) * cost_f)[0:max_pm], label='Total')
        # # plt.gca().set_ylim(bottom=None, top=treatment_strengths[max_pm]*cost_pm)
        # plt.ylabel('Cost')
        # plt.xlabel('PM')
        # plt.legend()
        # plt.show()

        # Optimal on average:
        pm_star_average_index = np.argmin(
            (treatment_strengths * cost_pm * max_t) + (o_pred.mean(axis=0) * cost_o) + (f_pred.mean(axis=0) * cost_f))
        print('Optimal number of PM (on average) - estimated: \t' + str(
            np.round(treatment_strengths[pm_star_average_index], 4)*max_t))
        print('Optimal number of PM (on average) - actual: \t' + str(np.round(treatment_strengths[np.argmin(
            (treatment_strengths * cost_pm * max_t) + (o_test.mean(axis=0) * cost_o) + (o_test.mean(axis=0) * cost_f))],
                                                                              4)*max_t))

        # Different policies: supervised individualized - causal individualized - causal average
        total_costs_pred_naive = np.tile(treatment_strengths, (
        x_test.shape[0], 1)) * cost_pm * max_t + o_pred_naive * cost_o + f_pred_naive * cost_f
        pm_star_pred_naive = np.take_along_axis(np.tile(treatment_strengths * max_t, (x_test.shape[0], 1)),
                                                np.argmin(total_costs_pred_naive, axis=1)[:, None], axis=1)

        total_costs_pred = np.tile(treatment_strengths,
                                   (x_test.shape[0], 1)) * cost_pm * max_t + o_pred * cost_o + f_pred * cost_f
        pm_star_pred = np.take_along_axis(np.tile(treatment_strengths * max_t, (x_test.shape[0], 1)),
                                          np.argmin(total_costs_pred, axis=1)[:, None], axis=1)

        total_costs_actual = np.tile(treatment_strengths,
                                     (x_test.shape[0], 1)) * cost_pm * max_t + o_test * cost_o + f_test * cost_f
        pm_star_ideal = np.take_along_axis(np.tile(treatment_strengths * max_t, (x_test.shape[0], 1)),
                                           np.argmin(total_costs_actual, axis=1)[:, None], axis=1)

        costs_policy_pred_naive = np.take_along_axis(total_costs_actual,
                                                     np.argmin(total_costs_pred_naive, axis=1)[:, None], axis=1)
        costs_policy_pred = np.take_along_axis(total_costs_actual,
                                               np.argmin(total_costs_pred, axis=1)[:, None], axis=1)
        costs_policy_average = total_costs_actual[:, total_costs_actual.mean(axis=0).argmin()]
        costs_policy_ideal = np.take_along_axis(total_costs_actual,
                                                np.argmin(total_costs_actual, axis=1)[:, None], axis=1)

        # Summarize
        print('\nPOLICY = INDIVIDUAL')
        print('Cost (number) of PM: \t\t' + str(np.round(
            np.take_along_axis(np.tile(treatment_strengths, (x_test.shape[0], 1)) * max_t,
                               np.argmin(total_costs_pred, axis=1)[:, None], axis=1).mean() * cost_pm, 2)) + ' (' + str(
            np.round(np.take_along_axis(np.tile(treatment_strengths, (x_test.shape[0], 1)) * max_t,
                                        np.argmin(total_costs_pred, axis=1)[:, None], axis=1).mean(), 2)) + ')')
        print('Cost (number) of Overhaul: \t' + str(
            np.round(np.take_along_axis(o_test, np.argmin(total_costs_pred, axis=1)[:, None], axis=1).mean() * cost_o,
                     2)) + ' (' + str(
            np.round(np.take_along_axis(o_test, np.argmin(total_costs_pred, axis=1)[:, None], axis=1).mean(),
                     2)) + ')')
        print('Cost (number) of Failure: \t' + str(
            np.round(np.take_along_axis(f_test, np.argmin(total_costs_pred, axis=1)[:, None], axis=1).mean() * cost_f,
                     2)) + ' (' + str(
            np.round(np.take_along_axis(f_test, np.argmin(total_costs_pred, axis=1)[:, None], axis=1).mean(),
                     2)) + ')')
        print('Average cost pred policy: \t' + str(np.round(costs_policy_pred.mean(), 2)))

        print('\nPOLICY = INDIVIDUAL (NAIVE)')
        print('Cost (number) of PM: \t\t' + str(np.round(
            np.take_along_axis(np.tile(treatment_strengths, (x_test.shape[0], 1)) * max_t,
                               np.argmin(total_costs_pred_naive, axis=1)[:, None], axis=1).mean() * cost_pm,
            2)) + ' (' + str(
            np.round(np.take_along_axis(np.tile(treatment_strengths, (x_test.shape[0], 1)) * max_t,
                                        np.argmin(total_costs_pred_naive, axis=1)[:, None], axis=1).mean(), 2)) + ')')
        print('Cost (number) of Overhaul: \t' + str(
            np.round(
                np.take_along_axis(o_test, np.argmin(total_costs_pred_naive, axis=1)[:, None], axis=1).mean() * cost_o,
                2)) + ' (' + str(
            np.round(np.take_along_axis(o_test, np.argmin(total_costs_pred_naive, axis=1)[:, None], axis=1).mean(),
                     2)) + ')')
        print('Cost (number) of Failure: \t' + str(
            np.round(
                np.take_along_axis(f_test, np.argmin(total_costs_pred_naive, axis=1)[:, None], axis=1).mean() * cost_f,
                2)) + ' (' + str(
            np.round(np.take_along_axis(f_test, np.argmin(total_costs_pred_naive, axis=1)[:, None], axis=1).mean(),
                     2)) + ')')
        print('Average cost pred policy: \t' + str(np.round(costs_policy_pred_naive.mean(), 2)))

        print('\nPOLICY = AVERAGE')
        print('Cost (number) of PM: \t\t' + str(
            np.round(treatment_strengths[pm_star_average_index] * cost_pm * max_t, 2)) + ' (' + str(
            np.round(treatment_strengths[pm_star_average_index] * max_t, 2)) + ')')
        print('Cost (number) of Overhaul: \t' + str(
            np.round((o_test[:, pm_star_average_index] * cost_o).mean(), 2)) + ' (' + str(
            np.round(o_test[:, pm_star_average_index].mean(), 2)) + ')')
        print('Cost (number) of Failure: \t' + str(
            np.round((f_test[:, pm_star_average_index] * cost_f).mean(), 2)) + ' (' + str(
            np.round(f_test[:, pm_star_average_index].mean(), 2)) + ')')
        print('Average cost avrg policy: \t' + str(np.round(costs_policy_average.mean(), 2)))

        print('\nPOLICY = IDEAL')
        # Todo: use actual predicted curve instead of point estimates! See SCIGAN
        print('Cost (number) of PM: \t\t' + str(
            np.round(np.take_along_axis(np.tile(treatment_strengths, (x_test.shape[0], 1)) * max_t,
                                        np.argmin(total_costs_actual, axis=1)[:, None], axis=1).mean() * cost_pm,
                     2)) + ' ('
              + str(np.round(np.take_along_axis(np.tile(treatment_strengths, (x_test.shape[0], 1)) * max_t,
                                                np.argmin(total_costs_actual, axis=1)[:, None], axis=1).mean(),
                             2)) + ')')
        print('Cost (number) of Overhaul: \t' + str(
            np.round(np.take_along_axis(o_test, np.argmin(total_costs_actual, axis=1)[:, None], axis=1).mean() * cost_o,
                     2)) + ' (' + str(
            np.round(np.take_along_axis(o_test, np.argmin(total_costs_actual, axis=1)[:, None], axis=1).mean(),
                     2)) + ')')
        print('Cost (number) of Failure: \t' + str(
            np.round(np.take_along_axis(f_test, np.argmin(total_costs_actual, axis=1)[:, None], axis=1).mean() * cost_f,
                     2)) + ' (' + str(
            np.round(np.take_along_axis(f_test, np.argmin(total_costs_actual, axis=1)[:, None], axis=1).mean(),
                     2)) + ')')
        print('Average cost ideal policy: \t' + str(np.round(costs_policy_ideal.mean(), 2)))

        print('\nComparison: ')
        pe_naive = np.mean(np.square(np.take_along_axis(np.tile(treatment_strengths, (x_test.shape[0], 1)) * max_t,
                                                        np.argmin(total_costs_actual, axis=1)[:, None],
                                                        axis=1) - np.take_along_axis(
            np.tile(treatment_strengths, (x_test.shape[0], 1)) * max_t,
            np.argmin(total_costs_pred_naive, axis=1)[:, None],
            axis=1)))
        pe_individual = np.mean(np.square(np.take_along_axis(np.tile(treatment_strengths, (x_test.shape[0], 1)) * max_t,
                                                             np.argmin(total_costs_actual, axis=1)[:, None],
                                                             axis=1) - np.take_along_axis(
            np.tile(treatment_strengths, (x_test.shape[0], 1)) * max_t, np.argmin(total_costs_pred, axis=1)[:, None],
            axis=1)))
        pe_average = np.mean(np.square(np.take_along_axis(np.tile(treatment_strengths, (x_test.shape[0], 1)) * max_t,
                                                          np.argmin(total_costs_actual, axis=1)[:, None], axis=1)
                                       - treatment_strengths[pm_star_average_index] * max_t))
        print('\nPolicy error - indiv: \t\t' + str(np.round(pe_individual, 4)))
        print('Policy error - naive: \t\t' + str(np.round(pe_naive, 4)))
        print('Policy error - avera: \t\t' + str(np.round(pe_average, 4)))

        # Compare errors:
        pes_mlp = np.take_along_axis(np.tile(treatment_strengths, (x_test.shape[0], 1)) * max_t, np.argmin(total_costs_actual, axis=1)[:, None], axis=1) \
                  - np.take_along_axis(np.tile(treatment_strengths, (x_test.shape[0], 1)) * max_t,
                                       np.argmin(total_costs_pred_naive, axis=1)[:, None], axis=1)
        pes_scigan = np.take_along_axis(np.tile(treatment_strengths, (x_test.shape[0], 1)) * max_t,
                                    np.argmin(total_costs_actual, axis=1)[:, None], axis=1) \
                 - np.take_along_axis(np.tile(treatment_strengths, (x_test.shape[0], 1)) * max_t,
                                      np.argmin(total_costs_pred, axis=1)[:, None], axis=1)
        pes_average = np.take_along_axis(np.tile(treatment_strengths, (x_test.shape[0], 1)) * max_t,
                                         np.argmin(total_costs_actual, axis=1)[:, None], axis=1) - treatment_strengths[
                          pm_star_average_index] * max_t

        pcr_naive = costs_policy_pred_naive.mean() / costs_policy_ideal.mean()
        pcr_individual = costs_policy_pred.mean() / costs_policy_ideal.mean()
        pcr_average = costs_policy_average.mean() / costs_policy_ideal.mean()
        print('\nAverage cost ratio - indiv: \t' + str(np.round(pcr_individual, 4)))
        print('Average cost ratio - naive: \t' + str(np.round(pcr_naive, 4)))
        print('Average cost ratio - avera: \t' + str(np.round(pcr_average, 4)))

        pcrs_mlp = costs_policy_pred_naive / costs_policy_ideal
        pcrs_scigan = costs_policy_pred / costs_policy_ideal
        pcrs_average = costs_policy_average[:, np.newaxis] / costs_policy_ideal

        plt.figure(figsize=(6, 3), dpi=400)
        sns.kdeplot(pes_scigan[:, 0], label="SCIGAN--ITE", linestyle=linestyles[0])
        sns.kdeplot(pes_mlp[:, 0], label="MLP--ITE", linestyle=linestyles[1])
        sns.kdeplot(pes_average[:, 0], label="SCIGAN--ATE", linestyle=linestyles[2])
        plt.legend(bbox_to_anchor=(1.1, 0.5), loc="center left", borderaxespad=0)
        plt.ylabel('')
        plt.xlabel('$t_i^* - \\hat{t}_i^*$')
        plt.savefig(DIR + 'Figures/Results_PEs_Iter_' + str(iter) + '.pdf')
        plt.show()

        plt.figure(figsize=(3, 3), dpi=400)
        sns.kdeplot(pcrs_scigan[:, 0], label="SCIGAN--ITE", linestyle=linestyles[0])
        sns.kdeplot(pcrs_mlp[:, 0], label="MLP--ITE", linestyle=linestyles[1])
        sns.kdeplot(pcrs_average[:, 0], label="SCIGAN--ATE", linestyle=linestyles[2])
        plt.legend()
        plt.ylabel('')
        plt.xlabel('PCR')
        plt.xlim([1, 2])
        plt.savefig(DIR + 'Figures/Results_PCRs_Iter_' + str(iter) + '.pdf')
        plt.show()

        fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(7, 3), dpi=1000)
        sns.kdeplot(pes_scigan[:, 0], label="SCIGAN--ITE", linestyle=linestyles[0], ax=ax0)
        sns.kdeplot(pes_mlp[:, 0], label="MLP--ITE", linestyle=linestyles[1], ax=ax0)
        sns.kdeplot(pes_average[:, 0], label="SCIGAN--ATE", linestyle=linestyles[2], ax=ax0)
        # ax0.legend(bbox_to_anchor=(1.1, 0.5), loc="center left", borderaxespad=0)
        ax0.set_ylabel('')
        ax0.set_xlabel('$t_i^* - \\hat{t}_i^*$')

        sns.kdeplot(pcrs_scigan[:, 0], label="SCIGAN--ITE", linestyle=linestyles[0], ax=ax1)
        sns.kdeplot(pcrs_mlp[:, 0], label="MLP--ITE", linestyle=linestyles[1], ax=ax1)
        sns.kdeplot(pcrs_average[:, 0], label="SCIGAN--ATE", linestyle=linestyles[2], ax=ax1)
        ax1.legend()
        # ax1.legend(bbox_to_anchor=(1.1, 0.5), loc="center left", borderaxespad=0)
        ax1.set_ylabel('')
        ax1.set_xlabel('PCR')
        ax1.set_xlim([1, 2])
        plt.savefig(DIR + 'Figures/Results_PEs_PCRs_Iter_ ' + str(iter) + '.pdf')
        plt.show()

        # Append results
        results_supervised['MISE_o'].append(mise_total_o_naive)
        results_supervised['MISE_f'].append(mise_total_f_naive)
        results_individual['MISE_o'].append(mise_total_o)
        results_individual['MISE_f'].append(mise_total_f)

        results_supervised['PE'].append(pe_naive)
        results_individual['PE'].append(pe_individual)
        results_average['PE'].append(pe_average)

        results_supervised['PCR'].append(costs_policy_pred_naive.mean() / costs_policy_ideal.mean())
        results_individual['PCR'].append(costs_policy_pred.mean() / costs_policy_ideal.mean())
        results_average['PCR'].append(costs_policy_average.mean() / costs_policy_ideal.mean())

    # Display results:
    print('\n------------------------------------------------')
    print('\nOVERALL RESULTS')

    print('\nMISE_O (INDIV): \t %.4f \t%.4f' % (
    np.round(np.mean(results_individual['MISE_o']), 4), np.round(np.std(results_individual['MISE_o']), 4)))
    print('MISE_O (NAIVE): \t %.4f \t%.4f' % (
    np.round(np.mean(results_supervised['MISE_o']), 4), np.round(np.std(results_supervised['MISE_o']), 4)))

    print('\nMISE_F (INDIV): \t %.4f \t%.4f' % (
    np.round(np.mean(results_individual['MISE_f']), 4), np.round(np.std(results_individual['MISE_f']), 4)))
    print('MISE_F (NAIVE): \t %.4f \t%.4f' % (
    np.round(np.mean(results_supervised['MISE_f']), 4), np.round(np.std(results_supervised['MISE_f']), 4)))

    # plt.figure(figsize=(3, 3))
    # plt.title('MISE Overhauls')
    # plt.boxplot([results_supervised['MISE_o'], results_individual['MISE_o']])
    # plt.gca().set_ylim(bottom=0, top=None)
    # plt.gca().set_xticks([y + 1 for y in range(2)], labels=['Supervised', 'Individual'])
    # plt.show()

    # plt.figure(figsize=(3, 3))
    # plt.title('MISE Failures')
    # plt.boxplot([results_supervised['MISE_f'], results_individual['MISE_f']])
    # plt.gca().set_ylim(bottom=0, top=None)
    # plt.gca().set_xticks([y + 1 for y in range(2)], labels=['Supervised', 'Individual'])
    # plt.show()

    print('\nPE (INDIV): \t\t %.4f \t%.4f' % (
        np.round(np.mean(results_individual['PE']), 4), np.round(np.std(results_individual['PE']), 4)))
    print('PE (NAIVE): \t\t %.4f \t%.4f' % (
        np.round(np.mean(results_supervised['PE']), 4), np.round(np.std(results_supervised['PE']), 4)))
    print('PE (AVERA): \t\t %.4f \t%.4f' % (
        np.round(np.mean(results_average['PE']), 4), np.round(np.std(results_average['PE']), 4)))

    # plt.figure(figsize=(3, 3))
    # plt.title('Policy error')
    # plt.boxplot([results_supervised['PE'], results_individual['PE'], results_average['PE']])
    # plt.gca().set_ylim(bottom=0, top=None)
    # plt.gca().set_xticks([y + 1 for y in range(3)], labels=['Supervised', 'Individual', 'Average'])
    # plt.show()

    print('\nPCR (INDIV): \t\t %.4f \t%.4f' % (
        np.round(np.mean(results_individual['PCR']), 4), np.round(np.std(results_individual['PCR']), 4)))
    print('PCR (NAIVE): \t\t %.4f \t%.4f' % (
        np.round(np.mean(results_supervised['PCR']), 4), np.round(np.std(results_supervised['PCR']), 4)))
    print('PCR (AVERA): \t\t %.4f \t%.4f' % (
        np.round(np.mean(results_average['PCR']), 4), np.round(np.std(results_average['PCR']), 4)))

    # plt.figure(figsize=(3, 3))
    # plt.title('Cost ratios')
    # plt.boxplot([results_supervised['PCR'], results_individual['PCR'], results_average['PCR']])
    # plt.gca().set_ylim(bottom=1, top=None)
    # plt.gca().set_xticks([y + 1 for y in range(3)], labels=['Supervised', 'Individual', 'Average'])
    # plt.show()
