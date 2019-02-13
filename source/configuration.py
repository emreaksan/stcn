import os
import json
import sys
import tensorflow as tf
from constants import Constants as C


class Configuration(object):
    """
    Base configuration class to setup experiments.
    """

    def __init__(self, **kwargs):
        self.config = kwargs
        self.data_dir = None
        self.preprocessing_ops = dict()

        # Set a seed value.
        if not ("seed" in self.config):
            self.config['seed'] = C.SEED
        self.seed = self.config.get('seed')
        tf.set_random_seed(self.seed)

    def override_data_path(self, training, validation=None):
        self.config['training_data'] = "{}{}".format(self.data_dir, training)
        if validation is not None:
            self.config['validation_data'] = "{}{}".format(self.data_dir, validation)

    def get_preprocessing_ops(self):
        self.preprocessing_ops[C.PP_SHIFT] = False
        self.preprocessing_ops[C.PP_ZERO_MEAN_NORM] = self.config.get(C.PP_ZERO_MEAN_NORM, False)
        self.preprocessing_ops[C.PP_ZERO_MEAN_NORM_SEQ] = self.config.get(C.PP_ZERO_MEAN_NORM_SEQ, False)
        self.preprocessing_ops[C.PP_ZERO_MEAN_NORM_ALL] = self.config.get(C.PP_ZERO_MEAN_NORM_ALL, False)

        return self.preprocessing_ops

    def get_sample_function(self):
        """
        Returns data-dependent sample construction functions (in tensorflow and numpy) given model outputs.
        For now we directly use model mean predictions.
        """
        raise Exception("Not implemented.")

    def get(self, param, default=None):
        """
        Get the value of the configuration parameter `param`. If `default` is set, this will be returned in case `param`
        does not exist. If `default` is not set and `param` does not exist, an error is thrown.
        """
        return self.config.get(param, default)

    def set(self, param, value, override=False):
        """
        Sets a new configuration parameter or overrides an existing one.
        """
        if not override and self.exists(param):
            raise RuntimeError('Key "{}" already exists. If you want to override set "override" to True.'.format(param))
        self.config[param] = value

    def exists(self, param):
        """
        Check if given configuration parameter exists.
        """
        return param in self.config.keys()

    def dump(self, path):
        """
        Stores this configuration object on disk in a human-readable format (json) and as a byte object (pickle).
        """
        json.dump(self.config, open(os.path.join(path, 'config.json'), 'w'), indent=4, sort_keys=True)

    @staticmethod
    def from_json(path):
        """
        Loads a configuration from json file.
        """
        return json.load(open(path, 'r'))

    @staticmethod
    def define_training_setup(parser):
        """
        Adds command line arguments for training script.

        Args:
            parser (argparse.ArgumentParser object):
        """
        parser.add_argument('--experiment_name', required=False, default=None, type=str, help='A short descriptive name to be used as prefix.')
        parser.add_argument('--training_data', required=True, type=str, help='Path to training dataset.')
        parser.add_argument('--validation_data', required=True, type=str, help='Path to validation dataset.')
        parser.add_argument('--test_data', required=False, default=None, type=str, help='Path to test dataset.')

        # Experiment outputs.
        parser.add_argument('--save_dir', type=str, default='./runs/', help='Path to main model save directory.')
        parser.add_argument('--eval_dir', type=str, help='Path to main log/output directory.')
        parser.add_argument('--checkpoint_id', type=str, default=None, help='Log and output directory.')
        parser.add_argument('--run_evaluation_after_training', action="store_true", required=False, help='Run evaluation after training.')
        # (1) To continue training.
        parser.add_argument('--model_id', required=False, default=None, type=str, help='10 digit experiment id.')
        # (2) To create a configuration object from json.
        parser.add_argument('--json_file', type=str, help='Path to a configuration saved as json file.')
        # Data preprocessing.
        parser.add_argument('--'+C.PP_ZERO_MEAN_NORM, action="store_true", help='Applies zero-mean unit-variance normalization.')
        parser.add_argument('--'+C.PP_ZERO_MEAN_NORM_SEQ, action="store_true", help='Applies zero-mean unit-variance normalization with sequence stats.')
        parser.add_argument('--'+C.PP_ZERO_MEAN_NORM_ALL, action="store_true", help='Applies zero-mean unit-variance normalization with stats calcualted by using all data entries.')

    @staticmethod
    def define_evaluation_setup(parser):
        """
        Adds command line arguments for evaluation script.

        Args:
            parser (argparse.ArgumentParser object):
        """
        # Experiment outputs.
        parser.add_argument('--save_dir', type=str, default='./runs/', help='path to main model save directory')
        parser.add_argument('--eval_dir', type=str, help='path to main log/output directory')
        parser.add_argument('--model_id', type=str, help='model folder', required=True)
        parser.add_argument('--checkpoint_id', type=str, default=None, help='Model checkpoint. If not set, then the last checkpoint is used.')

        parser.add_argument('--validation_data', required=False, type=str, help='Path to validation dataset.')
        parser.add_argument('--test_data', required=False, default=None, type=str, help='Path to test dataset.')

        # Analysis types.
        parser.add_argument('--quantitative', action="store_true", help='Run quantitative analysis.')
        parser.add_argument('--qualitative', action="store_true", help='Run qualitative analysis.')
        parser.add_argument('--verbose', dest='verbose', type=int, default=0, help='Verbosity of logs.')
        parser.add_argument('--seed', dest='seed', type=int, default=None, help='Seed value.')

    def set_experiment_name(self, use_template=True, experiment_name=None):
        """
        Creates a folder name based on data and model configuration.

        Args:
            use_template (bool): Whether to use data and model naming template.
            experiment_name (str): A descriptive experiment name. It is used as prefix if use_template is True.

        Returns:
            A descriptive string to be used for experiment folder name.
        """
        # TODO: this is messy and ugly.
        if use_template:
            str_loss = ""
            str_pp = ""
            if self.config.get(C.PP_ZERO_MEAN_NORM, False):
                str_pp += "_norm"
            if self.config.get(C.PP_ZERO_MEAN_NORM_SEQ, False):
                str_pp += "_snorm"
            if self.config.get(C.PP_ZERO_MEAN_NORM_ALL, False):
                str_pp += "_anorm"
            str_pp = str_pp[1:]

            if self.config['model_type'] in [C.MODEL_TCN, C.MODEL_STCN]:
                str_conv_block = ""
                if self.config['cnn_layer']['use_residual']:
                    str_conv_block += "-r"
                if self.config['cnn_layer']['use_skip']:
                    str_conv_block += "-s"
                if self.config.get('input_layer', None) is not None:
                    if self.config['input_layer'].get('dropout_rate', 0) > 0:
                        str_conv_block += "-idrop" + str(int(self.config['input_layer']['dropout_rate']*10))

                if self.config['model_type'] in [C.MODEL_STCN]:
                    if self.config['latent_layer']['type'] == C.LATENT_GAUSSIAN:
                        str_conv_block += "-vae_l" + str(self.config['latent_layer']['latent_size'])
                    elif self.config['latent_layer']['type'] == C.LATENT_LADDER_GAUSSIAN:
                        l_size_str = ""
                        """
                        latent_size = self.config['latent_layer']['latent_size']
                        if isinstance(latent_size, list):
                            if latent_size[1:] == latent_size[:-1]:  # Check if all entries are the same.
                                l_size_str = str(latent_size[0]) + "x" + str(len(latent_size))
                            else:
                                l_size_str = str(latent_size[0]) + "_pow"
                        else:
                            l_size_str = str(latent_size)
                        """
                        l_size_str += "_vd" + str(self.config['latent_layer']['vertical_dilation'])
                        l_size_str += "_dp" if self.config['latent_layer']['dynamic_prior'] else "_fp"
                        str_conv_block += "-ladderL" + l_size_str
                    # if self.config.get("num_latent_samples", 1) > 1:
                    #     str_conv_block += "-l_sample_" + str(self.config.get("num_latent_samples"))
                    if self.config['latent_layer']['layer_structure'] == C.LAYER_FC:
                        str_conv_block += "_Lfc_" + str(self.config['latent_layer']['num_hidden_units']) + "x" + str(self.config['latent_layer']['num_hidden_layers'])
                    elif self.config['latent_layer']['layer_structure'] == C.LAYER_TCN:
                        str_conv_block += "_Ltcn_" + str(self.config['latent_layer']['num_hidden_units']) + "x" + str(self.config['latent_layer']['num_hidden_layers'])
                        if self.config['latent_layer'].get('filter_size', 1) > 1:
                            str_conv_block += "_f" + str(self.config['latent_layer'].get('filter_size', 1))
                        if self.config['latent_layer'].get('dilation', 1) > 1:
                            str_conv_block += "_d" + str(self.config['latent_layer'].get('dilation', 1))
                    elif self.config['latent_layer']['layer_structure'] == C.LAYER_CONV1:
                        str_conv_block += "_Lconv1_" + str(self.config['latent_layer']['num_hidden_units']) + "x" + str(self.config['latent_layer']['num_hidden_layers'])
                    elif self.config['latent_layer']['layer_structure'] == C.LAYER_RNN:
                        str_conv_block += "_L" + str(self.config['latent_layer']['cell_type']) + "_" + str(self.config['latent_layer']['cell_num_layers']) + "x" + str(self.config['latent_layer']['cell_size'])

                    if self.config.get('decoder_use_enc_prev', False):
                        str_conv_block += "-prev_enc_dec"
                    if self.config.get('decoder_use_raw_inputs', False):
                        str_conv_block += "-raw_inp_dec"

                out_str = ""
                if self.config['output_layer']['type'] == C.LAYER_FC and self.config['output_layer']['num_layers'] > 0:
                    out_str = "-out_fc{}x{}".format(self.config['output_layer']['size'], self.config['output_layer']['num_layers'])
                elif self.config['output_layer']['type'] == C.LAYER_CONV1 and self.config['output_layer']['num_layers'] > 0:
                    out_size = self.config['cnn_layer']['num_filters'] if self.config['output_layer'].get('size', 0) < 1 else self.config['output_layer'].get('size')
                    out_str = "-out_conv1_{}x{}".format(out_size, self.config['output_layer']['num_layers'])
                elif self.config['output_layer']['type'] == C.LAYER_TCN and self.config['output_layer']['num_layers'] > 0:
                    out_size = self.config['cnn_layer']['num_filters'] if self.config['output_layer'].get('size', 0) < 1 else self.config['output_layer'].get('size')
                    kernel_size = self.config['cnn_layer']['filter_size'] if self.config['output_layer'].get('filter_size', 0) < 1 else self.config['output_layer'].get('filter_size', 0)
                    out_str = "-out_ccn_{}_{}x{}".format(out_size, kernel_size, self.config['output_layer']['num_layers'])

                if "num_encoder_layers" in self.config['cnn_layer']:
                    if self.config['cnn_layer']['num_decoder_layers'] > 0:
                        self.config['experiment_name'] = "{}_e{}_d{}_{}_{}{}{}-{}".format(self.config['model_type'], self.config['cnn_layer']['num_encoder_layers'], self.config['cnn_layer']['num_decoder_layers'], self.config['cnn_layer']['filter_size'], self.config['cnn_layer']['num_filters'], str_conv_block, out_str, self.config['cnn_layer']['activation_fn'])
                    else:
                        self.config['experiment_name'] = "{}_e{}_{}_{}{}{}-{}".format(self.config['model_type'],
                                                                                      self.config['cnn_layer']['num_encoder_layers'],
                                                                                      self.config['cnn_layer']['filter_size'],
                                                                                      self.config['cnn_layer']['num_filters'],
                                                                                      str_conv_block, out_str,
                                                                                      self.config['cnn_layer']['activation_fn'])
                else:
                    self.config['experiment_name'] = "{}{}_{}_{}{}{}-{}".format(self.config['model_type'],
                        self.config['cnn_layer']['num_layers'], self.config['cnn_layer']['filter_size'], self.config['cnn_layer']['num_filters'],
                        str_conv_block, out_str, self.config['cnn_layer']['activation_fn'])

            elif self.config['model_type'] in [C.MODEL_RNN]:
                str_drop = ""
                if self.config['input_layer'].get('dropout_rate', 0) > 0:
                    str_drop += "-idrop" + str(int(self.config['input_layer']['dropout_rate']*10))

                inp_fc_str = ""
                if self.config['input_layer']['num_layers'] > 0:
                    inp_fc_str = "-fc{}_{}".format(self.config['input_layer']['num_layers'], self.config['input_layer']['size'])
                out_fc_str = ""
                if self.config['output_layer']['num_layers'] > 0:
                    out_fc_str = "-fc{}_{}".format(self.config['output_layer']['num_layers'], self.config['output_layer']['size'])
                self.config['experiment_name'] = "{}{}-{}{}_{}{}{}-{}{}".format(self.config['model_type'],
                    inp_fc_str, self.config['rnn_layer']['cell_type'],
                    self.config['rnn_layer']['num_layers'], self.config['rnn_layer']['size'], out_fc_str, str_drop,
                    self.config['input_layer']['activation_fn'], str_loss)

            elif self.config['model_type'] == C.MODEL_VRNN:
                str_model = ""
                str_model += "-l" + str(self.config['latent_size'])
                str_model += "-h" + str(self.config['hidden_size'])
                str_model += "-fc" + str(self.config['num_fc_layers'])
                str_model += "-" + self.config['latent_rnn']['cell_type'] + "_" + str(
                    self.config['latent_rnn']['num_layers']) + "_" + str(self.config['latent_rnn']['size'])
                str_model += "-" + self.config['fc_layer_activation_func']
                self.config['experiment_name'] = "vrnn{}{}".format(str_model, str_loss)

            self.config['experiment_name'] = self.config['experiment_name'] + "-" + str_pp
        else:
            self.config['experiment_name'] = ""

        if experiment_name is not None and experiment_name is not "":
            self.config['experiment_name'] = experiment_name + "-" + self.config['experiment_name']

        return self.config['experiment_name']
