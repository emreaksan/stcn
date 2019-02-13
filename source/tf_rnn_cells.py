import tensorflow as tf

from tf_model_utils import linear, get_activation_fn, get_rnn_cell, fully_connected_layer
import tf_loss
from constants import Constants
C = Constants()


class VRNNCell(tf.contrib.rnn.RNNCell):
    """
    Variational RNN cell.

    Training time behaviour: draws latent vectors from approximate posterior distribution and tries to decrease the
    discrepancy between prior and the approximate posterior distributions.

    Sampling time behaviour: draws latent vectors from the prior distribution to synthesize a sample. This synthetic
    sample is then used to calculate approximate posterior distribution which is fed to RNN to update the state.
    The inputs to the forward call are not used and can be dummy.
    """

    def __init__(self, reuse, mode, sample_fn, config):
        """

        Args:
            reuse: reuse model parameters.
            mode: 'training' or 'sampling'.
            sample_fn: function to generate sample given model outputs.

            config (dict): In addition to standard <key, value> pairs, stores the following dictionaries for rnn and
                output configurations.

                config['output_layer'] = {}
                config['output_layer']['out_keys']
                config['output_layer']['out_dims']
                config['output_layer']['out_activation_fn']

                config['*_rnn'] = {}
                config['*_rnn']['num_layers'] (default: 1)
                config['*_rnn']['cell_type'] (default: lstm)
                config['*_rnn']['size'] (default: 512)
        """
        self.reuse = reuse
        self.mode = mode
        self.sample_fn = sample_fn
        self.is_sampling = mode == 'sampling'
        self.is_evaluation = mode == "validation" or mode == "test"

        self.input_dims = config['input_dims']
        self.h_dim = config['hidden_size']
        self.latent_h_dim = config.get('latent_hidden_size', self.h_dim)
        self.z_dim = config['latent_size']
        self.additive_q_mu = config['additive_q_mu']

        self.dropout_keep_prob = config.get('input_keep_prop', 1)
        self.num_linear_layers = config.get('num_fc_layers', 1)
        self.use_latent_h_in_outputs = config.get('use_latent_h_in_outputs', True)
        self.use_batch_norm = config['use_batch_norm_fc']

        if not (mode == "training"):
            self.dropout_keep_prob = 1.0

        self.output_config = config['output_layer']

        self.output_size_ = [self.z_dim]*4
        self.output_size_.extend(self.output_config['out_dims']) # q_mu, q_sigma, p_mu, p_sigma + model outputs

        self.state_size_ = []
        # Optional. Linear layers will be used if not passed.
        self.input_rnn = False
        if 'input_rnn' in config and not(config['input_rnn'] is None) and len(config['input_rnn'].keys()) > 0:
            self.input_rnn = True
            self.input_rnn_config = config['input_rnn']

            self.input_rnn_cell = get_rnn_cell(scope='input_rnn', **config['input_rnn'])

            # Variational dropout
            if config['input_rnn'].get('use_variational_dropout', False):
                # TODO input dimensions are hard-coded.
                self.input_rnn_cell = tf.contrib.rnn.DropoutWrapper(self.input_rnn_cell, input_keep_prob=self.dropout_keep_prob, output_keep_prob=self.dropout_keep_prob, variational_recurrent=True, input_size=(216), dtype=tf.float32)
                self.dropout_keep_prob = 1.0

            self.state_size_.append(self.input_rnn_cell.state_size)

        self.latent_rnn_config = config['latent_rnn']
        self.latent_rnn_cell_type = config['latent_rnn']['cell_type']
        self.latent_rnn_cell = get_rnn_cell(scope='latent_rnn', **config['latent_rnn'])
        self.state_size_.append(self.latent_rnn_cell.state_size)

        # Optional. Linear layers will be used if not passed.
        self.output_rnn = False
        if 'output_rnn' in config and not(config['output_rnn'] is None) and len(config['output_rnn'].keys()) > 0:
            self.output_rnn = True
            self.output_rnn_config = config['output_rnn']

            self.output_rnn_cell = get_rnn_cell(scope='output_rnn', **config['output_rnn'])
            self.state_size_.append(self.output_rnn_cell.state_size)

        self.activation_func = get_activation_fn(config.get('fc_layer_activation_func', 'relu'))
        self.sigma_activaction_fn = tf.nn.softplus

    @property
    def state_size(self):
        return tuple(self.state_size_)

    @property
    def output_size(self):
        return tuple(self.output_size_)

    #
    # Auxiliary functions
    #
    def draw_sample(self):
        """
        Draws a sample by using cell outputs.

        Returns:

        """
        return self.sample_fn(self.output_components)

    def reparametrization(self, mu, sigma, scope):
        """
        Given an isotropic normal distribution (mu and sigma), draws a sample by using reparametrization trick:
        z = mu + sigma*epsilon

        Args:
            mu: mean of isotropic Gaussian distribution.
            sigma: standard deviation of isotropic Gaussian distribution.

        Returns:

        """
        with tf.variable_scope(scope):
            eps = tf.random_normal(tf.shape(sigma), 0.0, 1.0, dtype=tf.float32)
            z = tf.add(mu, tf.multiply(sigma, eps))

            return z

    def phi(self, input_, scope, reuse=None):
        """
        A fully connected layer to increase model capacity and learn and intermediate representation. It is reported to
        be useful in https://arxiv.org/pdf/1506.02216.pdf

        Args:
            input_:
            scope:

        Returns:

        """
        with tf.variable_scope(scope, reuse=reuse):
            phi_hidden = input_
            for i in range(self.num_linear_layers):
                phi_hidden = linear(phi_hidden, self.h_dim, self.activation_func, batch_norm=self.use_batch_norm)

            return phi_hidden

    def latent(self, input_, scope):
        """
        Creates mu and sigma components of a latent distribution. Given an input layer, first applies a fully connected
        layer and then calculates mu & sigma.

        Args:
            input_:
            scope:

        Returns:

        """
        with tf.variable_scope(scope):
            latent_hidden = linear(input_, self.latent_h_dim, self.activation_func, batch_norm=self.use_batch_norm)
            with tf.variable_scope("mu"):
                mu = linear(latent_hidden, self.z_dim)
            with tf.variable_scope("sigma"):
                sigma = linear(latent_hidden, self.z_dim, self.sigma_activaction_fn)

            return mu, sigma

    def parse_rnn_state(self, state):
        """
        Sets self.latent_h and rnn states.

        Args:
            state:

        Returns:

        """
        latent_rnn_state_idx = 0
        if self.input_rnn is True:
            self.input_rnn_state = state[0]
            latent_rnn_state_idx = 1
        if self.output_rnn is True:
            self.output_rnn_state = state[latent_rnn_state_idx+1]

        # Check if the cell consists of multiple cells.
        self.latent_rnn_state = state[latent_rnn_state_idx]

        if self.latent_rnn_cell_type == C.GRU:
            self.latent_h = self.latent_rnn_state[-1] if type(self.latent_rnn_state) == tuple else self.latent_rnn_state
        else:
            self.latent_h = self.latent_rnn_state[-1].h if type(self.latent_rnn_state) == tuple else self.latent_rnn_state.h

    #
    # Functions to build graph.
    #
    def build_training_graph(self, input_, state):
        """

        Args:
            input_:
            state:

        Returns:

        """
        self.parse_rnn_state(state)
        self.input_layer(input_, state)
        self.input_layer_hidden()

        self.latent_p_layer()
        self.latent_q_layer()
        #if self.is_evaluation:
        #    self.phi_z = self.phi_z_p
        #else:
        self.phi_z = self.phi_z_q

        self.output_layer_hidden()
        self.output_layer()
        self.update_latent_rnn_layer()

    def build_sampling_graph(self, input_, state):
        self.parse_rnn_state(state)
        self.latent_p_layer()
        self.phi_z = self.phi_z_p

        self.output_layer_hidden()
        self.output_layer()

        # Draw a sample by using predictive distribution.
        synthetic_sample = self.draw_sample()
        # TODO: Is dropout required in `sampling` mode?
        self.input_layer(synthetic_sample, state)
        self.input_layer_hidden()
        self.latent_q_layer()
        self.update_latent_rnn_layer()


    def input_layer(self, input_, state):
        """
        Set self.x by applying dropout.
        Args:
            input_:
            state:

        Returns:

        """
        with tf.variable_scope("input"):
            input_components = tf.split(input_, self.input_dims, axis=1)
            self.x = input_components[0]

    def input_layer_hidden(self):
        if self.input_rnn is True:
            self.phi_x_input, self.input_rnn_state = self.input_rnn_cell(self.x, self.input_rnn_state, scope='phi_x_input')
        else:
            self.phi_x_input = self.phi(self.x, scope='phi_x_input')

        if self.dropout_keep_prob < 1.0:
            self.phi_x_input = tf.nn.dropout(self.phi_x_input, keep_prob=self.dropout_keep_prob)

    def latent_q_layer(self):
        input_latent_q = tf.concat((self.phi_x_input, self.latent_h), axis=1)
        if self.additive_q_mu:
            q_mu_delta, self.q_sigma = self.latent(input_latent_q, scope="latent_z_q")
            self.q_mu = q_mu_delta + self.p_mu
        else:
            self.q_mu, self.q_sigma = self.latent(input_latent_q, scope="latent_z_q")

        q_z = self.reparametrization(self.q_mu, self.q_sigma, scope="z_q")
        self.phi_z_q = self.phi(q_z, scope="phi_z", reuse=True)

    def latent_p_layer(self):
        input_latent_p = tf.concat((self.latent_h), axis=1)
        self.p_mu, self.p_sigma = self.latent(input_latent_p, scope="latent_z_p")

        p_z = self.reparametrization(self.p_mu, self.p_sigma, scope="z_p")
        self.phi_z_p = self.phi(p_z, scope="phi_z")

    def output_layer_hidden(self):
        if self.use_latent_h_in_outputs is True:
            output_layer_hidden = tf.concat((self.phi_z, self.latent_h), axis=1)
        else:
            output_layer_hidden = tf.concat((self.phi_z), axis=1)

        if self.output_rnn is True:
            self.phi_x_output, self.output_rnn_state = self.output_rnn_cell(output_layer_hidden, self.output_rnn_state, scope='phi_x_output')
        else:
            self.phi_x_output = self.phi(output_layer_hidden, scope="phi_x_output")

    def output_layer(self):
        self.output_components = {}
        for key, size, activation_func in zip(self.output_config['out_keys'], self.output_config['out_dims'], self.output_config['out_activation_fn']):
            with tf.variable_scope(key):
                if not callable(activation_func):
                    activation_func = get_activation_fn(activation_func)
                output_component = linear(self.phi_x_output, size, activation_fn=activation_func)
                self.output_components[key] = output_component

    def update_latent_rnn_layer(self):
        input_latent_rnn = tf.concat((self.phi_x_input, self.phi_z), axis=1)
        self.latent_rnn_output, self.latent_rnn_state = self.latent_rnn_cell(input_latent_rnn, self.latent_rnn_state)

    def __call__(self, input_, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__, reuse=self.reuse):
            if self.is_sampling:
                self.build_sampling_graph(input_, state)
            else:
                self.build_training_graph(input_, state)

            # Prepare cell output.
            vrnn_cell_output = [self.q_mu, self.q_sigma, self.p_mu, self.p_sigma]
            for key in self.output_config['out_keys']:
                vrnn_cell_output.append(self.output_components[key])

            # Prepare cell state.
            vrnn_cell_state = []
            if self.input_rnn:
                vrnn_cell_state.append(self.input_rnn_state)

            vrnn_cell_state.append(self.latent_rnn_state)

            if self.output_rnn:
                vrnn_cell_state.append(self.output_rnn_state)

            return tuple(vrnn_cell_output), tuple(vrnn_cell_state)