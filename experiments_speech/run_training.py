import tensorflow as tf
from tf_train import TrainingEngine
from constants import Constants as C
from configuration_speech import SpeechConfiguration as Configuration
from run_evaluation import do_evaluation

import argparse
import os
import glob


def run_training(config_obj, early_stopping_tolerance=10, run_evaluation_after_training=False):
    training_engine = TrainingEngine(config_obj, early_stopping_tolerance)
    training_engine.run()
    if config_obj.get("eval_dir", None) and run_evaluation_after_training:
        config_obj.set('eval_dir', os.path.join(config_obj.get("eval_dir"), config_obj.get('model_id')), override=True)
        if not os.path.exists(config_obj.get('eval_dir')):
            os.makedirs(config_obj.get('eval_dir'))
        config_obj.dump(config.get('eval_dir'))
        do_evaluation(config_obj, qualitative_analysis=True, quantitative_analysis=True, verbose=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    Configuration.define_training_setup(parser)

    args = parser.parse_args()
    args_dict = vars(args)

    if args.model_id is not None:
        # Restore
        model_dir = glob.glob(os.path.join(args.save_dir, "*tf-" + args.model_id + "-*"), recursive=False)[0]
        config_dict = Configuration.from_json(os.path.abspath(os.path.join(model_dir, 'config.json')))

        # In case the experiment folder is renamed, update the configuration.
        config_dict['model_dir'] = model_dir
    else:
        # Start a new experiment by loading a json configuration file.
        config_dict = Configuration.from_json(args.json_file)
        config_dict['model_dir'] = None

        assert args.training_data and os.path.exists(args.training_data), "Training data not found."
        assert args.validation_data and os.path.exists(args.validation_data), "Validation data not found."
        if args.test_data and not os.path.exists(args.test_data):
            raise Exception("Test data not found.")

    config = Configuration(**{**config_dict, **args_dict})
    config.set_experiment_name(experiment_name=args.experiment_name)
    run_training(config, run_evaluation_after_training=args.run_evaluation_after_training, early_stopping_tolerance=10)

