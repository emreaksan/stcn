import json
import numpy as np
import tensorflow as tf

from tf_dataset_ink import *
from tf_models import *
from configuration import Configuration
from constants import Constants as C


class InkConfiguration(Configuration):
    """
    Experiment configuration.
    """
    def __init__(self, **kwargs):
        super(InkConfiguration, self).__init__(**kwargs)

        self.model_cls = getattr(sys.modules[__name__], self.config.get('model_cls'))
        self.dataset_cls = getattr(sys.modules[__name__], self.config.get('dataset_cls'))

    def get_sample_function(self):
        """
        Returns data-dependent sample construction functions (in tensorflow and numpy) given model outputs.
        For now we directly use model mean predictions.
        """
        pen_threshold = 0.5

        def ink_sample_np(out_dict):
            if C.OUT_COEFFICIENT in out_dict:  # GMM model.
                raise Exception("Not implemented.")  # Numpy sampling function is not implemented for GMM.
            else:
                # stroke_sample = out_dict[C.OUT_MU]
                stroke_sample = np.random.normal(out_dict[C.OUT_MU], out_dict[C.OUT_SIGMA])
            pen = out_dict[C.OUT_BINARY]
            pen[np.where(pen > pen_threshold)] = 1.0
            pen[np.where(pen <= pen_threshold)] = 0.0
            return np.concatenate([stroke_sample, pen], axis=-1)

        def ink_sample_tf(out_dict):
            if C.OUT_COEFFICIENT in out_dict:  # GMM model.
                is_sequence = True if len(out_dict[C.OUT_MU].shape) == 3 else False
                mu_components = out_dict[C.OUT_MU] if is_sequence else tf.expand_dims(out_dict[C.OUT_MU], axis=1)
                sigma_components = out_dict[C.OUT_SIGMA] if is_sequence else tf.expand_dims(out_dict[C.OUT_SIGMA], axis=1)
                coefficients = out_dict[C.OUT_COEFFICIENT] if is_sequence else tf.expand_dims(out_dict[C.OUT_COEFFICIENT], axis=1)

                batch_size, seq_len, feature_gmm_components = mu_components.shape.as_list()
                _, _, num_gmm_components = coefficients.shape.as_list()
                feature_size = int(feature_gmm_components/num_gmm_components)
                seq_len = tf.shape(mu_components)[1] if seq_len is None else seq_len  # Variable-length sequences.
                batch_size = tf.shape(mu_components)[0] if batch_size is None else batch_size

                mu_ = tf.reshape(mu_components, (batch_size, seq_len, feature_size, num_gmm_components))
                sigma_ = tf.reshape(sigma_components, (batch_size, seq_len, feature_size, num_gmm_components))

                mu_ = tf.transpose(mu_, perm=[0, 1, 3, 2])
                sigma_ = tf.transpose(sigma_, perm=[0, 1, 3, 2])

                # Select the most likely mixture component.
                probs = tf.reshape(coefficients, (-1, num_gmm_components))
                logits = tf.log(probs) + 1.0
                component_indices = tf.reshape(tf.multinomial(logits, 1, seed=self.seed, name="gmm_component"), (batch_size, seq_len))

                batch_indices = tf.range(batch_size)
                seq_indices = tf.range(seq_len)

                idx_grid = tf.meshgrid(batch_indices, seq_indices)
                gather_idx = tf.stack([tf.transpose(idx_grid[0]), tf.transpose(idx_grid[1]), tf.cast(component_indices, tf.int32)], axis=-1)
                component_mu = tf.gather_nd(mu_, gather_idx)
                component_sigma = tf.gather_nd(sigma_, gather_idx)

                stroke_sample = tf.random_normal(tf.shape(component_mu), component_mu, component_sigma)
                if is_sequence is False:
                    stroke_sample = stroke_sample[:, 0]  # Ignore the sequence dimension.
            else:
                stroke_sample = tf.random_normal(tf.shape(out_dict[C.OUT_MU]), out_dict[C.OUT_MU], out_dict[C.OUT_SIGMA])

            pen = out_dict[C.OUT_BINARY]
            pen = tf.where(tf.greater(pen, tf.fill(tf.shape(pen), pen_threshold)), tf.fill(tf.shape(pen), 1.0), tf.fill(tf.shape(pen), 0.0))
            return tf.concat([stroke_sample, pen], axis=-1)

        return ink_sample_tf, ink_sample_np

    def get_preprocessing_ops(self):
        super(InkConfiguration, self).get_preprocessing_ops()
        self.preprocessing_ops[C.PP_SHIFT] = self.config.get('model_type') in [C.MODEL_RNN, C.MODEL_TCN]

        return self.preprocessing_ops

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
        Configuration.define_training_setup(parser)

    @staticmethod
    def define_evaluation_setup(parser):
        """
        Adds command line arguments for evaluation script.

        Args:
            parser (argparse.ArgumentParser object):
        """
        Configuration.define_evaluation_setup(parser)

        parser.add_argument('--pad_original', type=int, default=0, help='Concatenate synthetic sample with provided number of seed sample frames.')