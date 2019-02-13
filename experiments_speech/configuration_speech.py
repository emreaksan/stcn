import json
from configuration import Configuration

from tf_dataset import *
from tf_models import *

from constants import Constants
C = Constants()


class SpeechConfiguration(Configuration):
    """
    Experiment configuration.
    """

    def __init__(self, **kwargs):
        super(SpeechConfiguration, self).__init__(**kwargs)

        self.model_cls = getattr(sys.modules[__name__], self.config.get('model_cls'))
        self.dataset_cls = getattr(sys.modules[__name__], self.config.get('dataset_cls'))

    def get_sample_function(self):
        """
        Returns data-dependent sample construction functions (in tensorflow and numpy) given model outputs.
        For now we directly use model mean predictions.
        """
        def sample_np(out_dict):
            return out_dict[C.OUT_MU]

        def sample_tf(out_dict):
            return out_dict[C.OUT_MU]

        return sample_tf, sample_np

    def get_preprocessing_ops(self):
        super(SpeechConfiguration, self).get_preprocessing_ops()
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