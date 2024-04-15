# Adapted from SCIGAN, see https://github.com/ioanabica/SCIGAN/blob/main/SCIGAN.py


import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tqdm import tqdm
import os

from src.utils.model_utils import equivariant_layer, invariant_layer, sample_dosages, sample_X, sample_Z

class SCIGAN_Supervised:
    def __init__(self, params):
        self.num_features = params['num_features']
        self.num_treatments = params['num_treatments']
        self.export_dir = params['export_dir']

        self.h_dim = params['h_dim']
        self.h_inv_eqv_dim = params['h_inv_eqv_dim']
        self.batch_size = params['batch_size']
        self.alpha = params['alpha']
        self.num_dosage_samples = params['num_dosage_samples']

        self.size_z = self.num_treatments * self.num_dosage_samples
        self.num_outcomes = self.num_treatments * self.num_dosage_samples

        tf.reset_default_graph()
        tf.random.set_random_seed(10)

        # Feature (X)
        self.X = tf.placeholder(tf.float32, shape=[None, self.num_features], name='input_features')
        # Treatment (T) - one-hot encoding for the treatment
        self.T = tf.placeholder(tf.float32, shape=[None, self.num_treatments], name='input_treatment')
        # Dosage (D)
        self.D = tf.placeholder(tf.float32, shape=[None, 1], name='input_dosage')
        # Dosage samples (D)
        self.Treatment_Dosage_Samples = tf.placeholder(tf.float32,
                                                       shape=[None, self.num_treatments, self.num_dosage_samples],
                                                       name='input_treatment_dosage_samples')
        # Treatment dosage mask to indicate the factual outcome
        self.Treatment_Dosage_Mask = tf.placeholder(tf.float32,
                                                    shape=[None, self.num_treatments, self.num_dosage_samples],
                                                    name='input_treatment_dosage_mask')
        # Outcome (Y)
        self.Y = tf.placeholder(tf.float32, shape=[None, 1], name='input_y')
        # Random Noise (G)
        self.Z_G = tf.placeholder(tf.float32, shape=[None, self.size_z], name='input_noise')

    def inference(self, x, treatment_dosage_samples):
        with tf.variable_scope('inference', reuse=tf.AUTO_REUSE):
            inputs = x
            I_shared = tf.layers.dense(inputs, self.h_dim, activation=tf.nn.elu, name='shared')

            I_treatment_dosage_outcomes = dict()

            for treatment in range(self.num_treatments):
                dosage_counterfactuals = dict()
                treatment_dosages = treatment_dosage_samples[:, treatment]

                for index in range(self.num_dosage_samples):
                    dosage_sample = tf.expand_dims(treatment_dosages[:, index], axis=-1)
                    input_counterfactual_dosage = tf.concat(axis=1, values=[I_shared, dosage_sample])

                    treatment_layer_1 = tf.layers.dense(input_counterfactual_dosage, self.h_dim, activation=tf.nn.elu,
                                                        name='treatment_layer_1_%s' % str(treatment),
                                                        reuse=tf.AUTO_REUSE)

                    treatment_layer_2 = tf.layers.dense(treatment_layer_1, self.h_dim, activation=tf.nn.elu,
                                                        name='treatment_layer_2_%s' % str(treatment),
                                                        reuse=tf.AUTO_REUSE)

                    treatment_dosage_output = tf.layers.dense(treatment_layer_2, 1, activation=None,
                                                              name='treatment_output_%s' % str(treatment),
                                                              reuse=tf.AUTO_REUSE)

                    # treatment_dosage_output = tf.layers.dense(treatment_layer_2, 1, activation=tf.nn.relu,
                    #                                           name='treatment_output_%s' % str(treatment),
                    #                                           reuse=tf.AUTO_REUSE)

                    dosage_counterfactuals[index] = treatment_dosage_output

                I_treatment_dosage_outcomes[treatment] = tf.concat(list(dosage_counterfactuals.values()), axis=-1)

            I_logits = tf.concat(list(I_treatment_dosage_outcomes.values()), axis=1)
            I_logits = tf.reshape(I_logits, shape=(-1, self.num_treatments, self.num_dosage_samples))

        return I_logits, I_treatment_dosage_outcomes

    def train(self, Train_X, Train_T, Train_D, Train_Y, verbose=False):

        # 4. Inference network
        I_logits, I_treatment_dosage_outcomes = self.inference(self.X, self.Treatment_Dosage_Samples)

        I_outcomes = tf.identity(I_logits, name="inference_outcomes")

        # 4. Inference loss
        I_logit_factual = tf.expand_dims(tf.reduce_sum(self.Treatment_Dosage_Mask * I_logits, axis=[1, 2]), axis=-1)
        # I_loss1 = tf.reduce_mean((G_logits - I_logits) ** 2)
        I_loss2 = tf.reduce_mean((self.Y - I_logit_factual) ** 2)
        # I_loss = tf.sqrt(I_loss1) + tf.sqrt(I_loss2)
        I_loss = tf.sqrt(I_loss2)

        # theta_G = tf.trainable_variables(scope='generator')
        # theta_D_dosage = tf.trainable_variables(scope='dosage_discriminator')
        # theta_D_treatment = tf.trainable_variables(scope='treatment_discriminator')
        theta_I = tf.trainable_variables(scope='inference')

        # %% Solver
        # G_solver = tf.train.AdamOptimizer(learning_rate=0.001).minimize(G_loss, var_list=theta_G)
        # D_dosage_solver = tf.train.AdamOptimizer(learning_rate=0.001).minimize(D_dosage_loss, var_list=theta_D_dosage)
        # D_treatment_solver = tf.train.AdamOptimizer(learning_rate=0.001).minimize(D_treatment_loss,
        #                                                                           var_list=theta_D_treatment)
        I_solver = tf.train.AdamOptimizer(learning_rate=0.001).minimize(I_loss, var_list=theta_I)

        # Setup tensorflow
        # print('TF Device' + str(tf.test.gpu_device_name()))
        # if tf.test.is_gpu_available():
        #     tf_device = 'gpu'
        # else:
        #     tf_device = 'cpu'
        tf_device = 'cpu'       # Slower anyway
        if tf_device == "cpu":
            tf_config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 0})
        else:
            tf_config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 1})
            tf_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=tf_config)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        # Train Inference Network
        print("Training inference network.")
        for it in tqdm(range(10000)):
            idx_mb = sample_X(Train_X, self.batch_size)
            X_mb = Train_X[idx_mb, :]
            T_mb = np.reshape(Train_T[idx_mb], [self.batch_size, ])
            D_mb = np.reshape(Train_D[idx_mb], [self.batch_size, ])
            Y_mb = np.reshape(Train_Y[idx_mb], [self.batch_size, 1])
            Z_G_mb = sample_Z(self.batch_size, self.size_z)

            treatment_dosage_samples = sample_dosages(self.batch_size, self.num_treatments, self.num_dosage_samples)
            factual_dosage_position = np.random.randint(self.num_dosage_samples, size=[self.batch_size])
            treatment_dosage_samples[range(self.batch_size), T_mb, factual_dosage_position] = D_mb

            treatment_dosage_mask = np.zeros(shape=[self.batch_size, self.num_treatments,
                                                    self.num_dosage_samples])
            treatment_dosage_mask[range(self.batch_size), T_mb, factual_dosage_position] = 1
            treatment_one_hot = np.sum(treatment_dosage_mask, axis=-1)

            _, I_loss_curr = self.sess.run([I_solver, I_loss],
                                           feed_dict={self.X: X_mb, self.T: treatment_one_hot,
                                                      self.D: D_mb[:, np.newaxis],
                                                      self.Treatment_Dosage_Samples: treatment_dosage_samples,
                                                      self.Treatment_Dosage_Mask: treatment_dosage_mask, self.Y: Y_mb,
                                                      self.Z_G: Z_G_mb})

            # %% Debugging
            if it % 1000 == 0 and verbose:
                print('Iter: {}'.format(it))
                print('I_loss: {:.4}'.format((I_loss_curr)))
                print()

        self.I_logits = I_logits

        tf.compat.v1.saved_model.simple_save(self.sess, export_dir=self.export_dir,
                                             inputs={'input_features': self.X,
                                                     'input_treatment_dosage_samples': self.Treatment_Dosage_Samples},
                                             outputs={'inference_outcome': I_logits})

    def tune(self, x_train, t_train, y_train, x_val, t_val, y_val, batch_sizes, h_dims):

        best_val_mise = np.inf

        # Set parameters:
        params = dict()
        params['num_features'] = self.num_features
        params['num_treatments'] = self.num_treatments
        params['export_dir'] = self.export_dir

        params['h_inv_eqv_dim'] = self.h_inv_eqv_dim
        params['alpha'] = self.alpha
        params['num_dosage_samples'] = self.num_dosage_samples

        for batch_size in batch_sizes:
            params['batch_size'] = batch_size

            for h_dim in h_dims:
                params['h_dim'] = h_dim

                model = SCIGAN_Supervised(params)
                model.train(Train_X=x_train, Train_T=np.zeros(len(x_train)).astype(int), Train_D=t_train,
                              Train_Y=y_train, verbose=False)

                # Calculate validation MISE (observed):
                with tf.Session(graph=tf.Graph()) as sess:
                    tf.saved_model.loader.load(sess, ["serve"], '')

                    pred = sess.run('inference_outcomes:0',
                                        feed_dict={'input_features:0': x_val,
                                                   'input_treatment_dosage_samples:0': np.tile(t_val, self.num_dosage_samples)[:, np.newaxis, :]})

                    y_pred = pred[:, 0, 0]

                    val_mise = np.mean(np.square(y_pred - y_val))
                    print('Parameters: ')
                    print('Batch size: ' + str(batch_size) + '\tHidden dim: ' + str(h_dim))
                    print('MSE Validation \t' + str(np.round(val_mise, 4)))

                    if val_mise < best_val_mise:
                        print('New best! -------------------------------------------------------------')

                        best_val_mise = val_mise

                        self.batch_size = batch_size
                        self.h_dim = h_dim

                        # Save model:
                        wdir = os.getcwd()
                        os.chdir(wdir + '/Best_Supervised')
                        tf.compat.v1.saved_model.simple_save(sess, export_dir=model.export_dir,
                                                             inputs={'input_features': model.X,
                                                                     'input_treatment_dosage_samples': model.Treatment_Dosage_Samples},
                                                             outputs={'inference_outcome': model.I_logits})
                        os.chdir(wdir)

        print('\nBest values: ==========================================================')
        print('MSE: ' + str(np.round(best_val_mise, 4)))
        print('Batch size: ' + str(self.batch_size))
        print('Hidden dim: ' + str(self.h_dim))
