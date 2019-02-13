import tensorflow as tf
import numpy as np

import sys
import os
import argparse
import glob
import time
import math
import warnings

from tf_dataset import *
from tf_models import *
from tf_data_feeder import DataFeederTF
from loss import kld_normal_isotropic
from configuration_speech import SpeechConfiguration as Configuration

import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


def draw_line_plot_matplotlib(x_data, y_data, idx, legend, filename):
    fig = plt.figure(1, figsize=(10, 7))
    if isinstance(idx, list):  # Multiple lines.
        for i in idx:
            plt.plot(x_data[i], y_data[i], label=legend[i], linewidth=1.5, color=C.RGB_COLORS[i]/255)
    else:
        plt.plot(x_data, y_data, linewidth=1.5, color=C.RGB_COLORS[idx]/255)
    plt.legend(legend)
    plt.grid(color=(239/255., 239/255., 239/255.))  # very light gray
    plt.savefig(filename, format="svg")
    plt.close(fig)


def plots_ladder_latent_variables(result_dict, filename, plot_q_mu_diff=False, print_latent_stats=False):
    # q_mu (min, max), (mean, std)
    stats_summary_format = "{} ({:.3f}, {:.3f}), ({:.3f}, {:.3f}) \n"
    num_components = len(result_dict["q_dists"])
    all_kld_loss_txt = ""

    all_x_data = []
    all_y_data = []
    all_legend = []
    all_idx = []

    all_x_data_q_diff = []
    all_y_data_q_diff = []
    all_legend_q_diff = []

    # Plot individual terms.
    for idx in range(num_components):
        kld_loss = result_dict["sequence_kld_" + str(idx)]
        q_dist_mu, q_dist_sigma = result_dict["q_dists"][idx]
        p_dist_mu, p_dist_sigma = result_dict["p_dists"][idx]
        name = "z" + str(idx)

        x_data = np.linspace(1, kld_loss.shape[1], kld_loss.shape[1])
        y_data = kld_loss[0, :, 0]
        draw_line_plot_matplotlib(x_data, y_data, idx, [name], filename+"_plot_kld_" + name)

        all_x_data.append(x_data)
        all_y_data.append(y_data)
        all_legend.append(name)
        all_idx.append(idx)

        kld_loss_txt = "KLD #" + str(idx) + ": " + str(result_dict["summary_kld_" + str(idx)])
        all_kld_loss_txt += kld_loss_txt + "\n"

        if print_latent_stats:
            all_kld_loss_txt += stats_summary_format.format("q_mu", q_dist_mu.min(), q_dist_mu.max(), q_dist_mu.mean(), q_dist_mu.std())
            all_kld_loss_txt += stats_summary_format.format("p_mu", p_dist_mu.min(), p_dist_mu.max(), p_dist_mu.mean(), p_dist_mu.std())

            all_kld_loss_txt += stats_summary_format.format("q_std", q_dist_sigma.min(), q_dist_sigma.max(), q_dist_sigma.mean(), q_dist_sigma.std())
            all_kld_loss_txt += stats_summary_format.format("p_std", p_dist_sigma.min(), p_dist_sigma.max(), p_dist_sigma.mean(), p_dist_sigma.std())

        if plot_q_mu_diff:
            y_data_q_diff = np.square(q_dist_mu[0][1:] - q_dist_mu[0][:-1]).sum(axis=1)
            name = "q_mu_" + str(idx)
            draw_line_plot_matplotlib(x_data[:-1], y_data_q_diff, idx, [name], filename + "_diff_" + name)
            all_y_data_q_diff.append(y_data_q_diff)
            all_x_data_q_diff.append(x_data[:-1])
            all_legend_q_diff.append(name)
            all_kld_loss_txt += "Q_MU_DIFF #" + str(idx) + ": " + str(y_data_q_diff.mean()) + "\n"

        if print_latent_stats or plot_q_mu_diff:
            all_kld_loss_txt += "\n"

    # Plot all terms together.
    draw_line_plot_matplotlib(all_x_data, all_y_data, all_idx, all_legend, filename+"_plot_kld_all")
    if plot_q_mu_diff:
        draw_line_plot_matplotlib(all_x_data_q_diff, all_y_data_q_diff, all_idx, all_legend_q_diff, filename + "_diff_q_mu_all")

    # Print KLD loss and stats.
    print(all_kld_loss_txt)
    with open(filename + "_kld_loss.txt", "w") as f:
        f.write(all_kld_loss_txt)


def visualize_audio_samples(samples, ids, output_path, color_labels=None):
    """
    A function to plot/visualize/save speech samples. Not implemented.
    Args:
        samples:
        ids:
        output_path:
        color_labels:
    Returns:
    """
    pass


def do_evaluation(config_obj, qualitative_analysis=True, quantitative_analysis=True, verbose=0):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    Model_cls = config_obj.model_cls
    Dataset_cls = config_obj.dataset_cls
    seed = config_obj.get("seed")

    biased_sampling_steps = -1  # Use all time-steps of the real data sample in biased synthesis.
    if config_obj.get('data_type') == C.TIMIT:
        batch_size = 32
        config_obj.config['reduce_loss'] = C.R_MEAN_SEQUENCE
        run_biased_synthesis = False
        run_synthesis = False
        run_reconstruction = False
        run_original_sample = False
        run_latent_variable_plots = False
        sample_ids = [100]
        plot_sample_ids = [100]
        synthetic_sample_length = 1000
    elif config_obj.get('data_type') == C.BLIZZARD:
        batch_size = 32
        config_obj.config['reduce_loss'] = C.R_MEAN_SEQUENCE
        run_biased_synthesis = False
        run_synthesis = False
        run_reconstruction = False
        run_original_sample = False
        run_latent_variable_plots = False
        sample_ids = [100]
        plot_sample_ids = [100]
        synthetic_sample_length = 1000
    else:
        raise Exception("Unknown data type.")

    _, sample_fn = config_obj.get_sample_function()

    # Data preprocessing configuration.
    preprocessing_ops = config_obj.get_preprocessing_ops()

    if config_obj.get('test_data', None) is not None:
        eval_data_path = config_obj.get('test_data')
        print("Loading test split...")
    else:
        print("!!!Test split is not found in config. Evaluating on the validation split.!!!")
        print("Loading validation split...")
        eval_data_path = config_obj.get('validation_data')

    evaluation_dataset = Dataset_cls(eval_data_path, preprocessing_ops=preprocessing_ops, var_len_seq=True)
    num_validation_iterations = math.ceil(evaluation_dataset.num_samples/batch_size)

    # Only use 1 queue thread, otherwise validation loop it gets blocked.
    valid_data_feeder = DataFeederTF(evaluation_dataset, 1, batch_size, queue_capacity=1024, shuffle=False, allow_smaller_final_batch=True)
    data_placeholders = valid_data_feeder.batch_queue(dynamic_pad=evaluation_dataset.is_dynamic, queue_capacity=512, queue_threads=4)

    # Create a session object and initialize parameters.
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

    if quantitative_analysis:
        # Start filling the queues.
        # Run model on validation data an report performance under the metric used for training.
        coord = tf.train.Coordinator()
        if config_obj.get('validate_model', False):
            valid_data_feeder.init(sess, coord)
        queue_threads = tf.train.start_queue_runners(coord=coord, sess=sess, start=True)
        # queue_threads.append(valid_data_feeder.enqueue_threads)

    print("Building the Network...")
    with tf.name_scope("validation"):
        validation_model = Model_cls(config=config_obj,
                                     session=sess,
                                     reuse=False,
                                     mode=C.EVAL,
                                     placeholders=data_placeholders,
                                     input_dims=evaluation_dataset.input_dims,
                                     target_dims=evaluation_dataset.target_dims, )
        validation_model.build_graph()
        validation_model.ops_for_eval_mode[C.OUT_MU] = tf.nn.sigmoid(validation_model.ops_model_output[C.OUT_MU])

    if qualitative_analysis:
        with tf.name_scope("sampling"):
            sampling_model = Model_cls(config=config_obj,
                                       session=sess,
                                       reuse=True,
                                       mode=C.SAMPLE,
                                       placeholders=data_placeholders,
                                       input_dims=evaluation_dataset.input_dims,
                                       target_dims=evaluation_dataset.target_dims, )
            sampling_model.build_graph()
            sampling_model.ops_evaluation[C.OUT_MU] = tf.nn.sigmoid(sampling_model.ops_model_output[C.OUT_MU])

    # Restore computation graph.
    try:
        saver = tf.train.Saver()
        # Restore variables.
        if config_obj.get('checkpoint_id') is None:
            checkpoint_path = tf.train.latest_checkpoint(config_obj.get('model_dir'))
        else:
            checkpoint_path = os.path.join(config_obj.get('model_dir'), config_obj.get('checkpoint_id'))

        print("Loading model " + checkpoint_path)
        saver.restore(sess, checkpoint_path)
    except:
        raise Exception("Model is not found.")

    if quantitative_analysis:
        print("Calculating likelihood...")
        # Get final validation error.
        valid_summary, valid_eval_loss = validation_model.evaluation_step_test_time(coord, queue_threads, 1, 1, num_validation_iterations)
        try:
            sess.run(valid_data_feeder.input_queue.close(cancel_pending_enqueues=True))
            coord.request_stop()
            coord.join(queue_threads, stop_grace_period_secs=5)
        except:
            pass

    if qualitative_analysis:
        print("Generating samples...")
        for no, sample_id in enumerate(sample_ids):
            # Fetch a sample
            _, validation_sample, validation_target = evaluation_dataset.fetch_sample(sample_id)
            # Prepare the sample and its reconstruction for visualization.
            if run_original_sample:
                original_sample = evaluation_dataset.prepare_for_visualization(validation_sample)
                out_path = os.path.join(config_obj.get('eval_dir'), "real_")
                visualize_audio_samples(original_sample, [sample_id], out_path)

            if run_reconstruction or run_latent_variable_plots:
                output_dict = validation_model.reconstruct(input_sequence=validation_sample, target_sequence=validation_target, use_sample_mean=True)

                if run_latent_variable_plots and sample_id in plot_sample_ids and "eval_dict" in output_dict:
                    out_path = os.path.join(config_obj.get('eval_dir'), "s" + str(sample_id))
                    plots_ladder_latent_variables(output_dict["eval_dict"], out_path, plot_q_mu_diff=True, print_latent_stats=True)
                if run_reconstruction:
                    reconstructed_sample = output_dict['sample']
                    out_path = os.path.join(config_obj.get('eval_dir'), "reconstructed_seed" + str(seed) + "_")
                    visualize_audio_samples(reconstructed_sample, [sample_id], out_path)

            if run_biased_synthesis:
                biased_sample_inputs = validation_sample[:, 0:biased_sampling_steps]
                biased_sample_targets = validation_target[:, 0:biased_sampling_steps]
                # biased_sampling_steps many steps are copied from the real sample. The rest is completed by the model.
                biased_sample_length = synthetic_sample_length - biased_sampling_steps if biased_sampling_steps > 0 else synthetic_sample_length
                if config_obj.get("model_type") in [C.MODEL_VRNN]:
                    output_dict = validation_model.reconstruct(input_sequence=biased_sample_inputs, target_sequence=biased_sample_targets, use_sample_mean=True)
                    output_dict = sampling_model.sample(seed_state=output_dict['state'], sample_length=biased_sample_length, use_sample_mean=True)
                    output_dict['sample'] = np.expand_dims(output_dict['sample'], axis=0)
                else:
                    output_dict = sampling_model.sample(seed_sequence=biased_sample_inputs, sample_length=biased_sample_length, use_sample_mean=True)

                # Concatenate synthetic sample with the original one.
                synthetic_sample_shape = output_dict['sample'].shape
                synthetic_sample = np.concatenate([biased_sample_inputs, output_dict['sample']], axis=1)
                original_sample_shape = biased_sample_inputs.shape
                color_labels = np.concatenate([np.ones((original_sample_shape[0], original_sample_shape[1])), np.ones((synthetic_sample_shape[0], synthetic_sample_shape[1]))*2], axis=1)

                # synthetic_sample = evaluation_dataset.prepare_for_visualization(synthetic_sample)
                out_path = os.path.join(config_obj.get('eval_dir'), "synthetic_biased_seed" + str(seed) + "_")
                visualize_audio_samples(synthetic_sample, [sample_id], out_path, color_labels=color_labels)

            if run_synthesis:
                if config_obj.get("model_type") in [C.MODEL_VRNN]:
                    output_dict = sampling_model.sample(seed_state=None, sample_length=synthetic_sample_length, use_sample_mean=True)
                    output_dict['sample'] = np.expand_dims(output_dict['sample'], axis=0)
                else:
                    output_dict = sampling_model.sample(seed_sequence=validation_sample[:, 0:1, :], sample_length=synthetic_sample_length, use_sample_mean=True)

                synthetic_sample = output_dict['sample']

                # synthetic_sample = evaluation_dataset.prepare_for_visualization(synthetic_sample)
                out_path = os.path.join(config_obj.get('eval_dir'), "synthetic_seed" + str(seed) + "_")
                visualize_audio_samples(synthetic_sample, [no], out_path)

    sess.close()
    tf.reset_default_graph()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    Configuration.define_evaluation_setup(parser)
    args = parser.parse_args()

    try:
        model_dir = glob.glob(os.path.join(args.save_dir, "*tf-" + args.model_id + "-*"), recursive=False)[0]
    except IndexError:
        raise Exception("Model " + str(args.model_id) + " is not found in " + str(args.save_dir))

    config_dict = Configuration.from_json(os.path.abspath(os.path.join(model_dir, 'config.json')))
    if args.seed is not None:
        config_dict["seed"] = args.seed

    # If validation_data or test_data not passed, the value in the config.json will be used.
    if args.validation_data and os.path.exists(args.validation_data):
        assert os.path.exists(args.validation_data), "Validation data not found."
        config_dict["validation_data"] = args.validation_data
    if args.test_data and os.path.exists(args.test_data):
        assert os.path.exists(args.test_data), "Test data not found."
        config_dict["test_data"] = args.test_data

    config = Configuration(**config_dict)
    config.set('checkpoint_id', args.checkpoint_id, override=True)
    if args.eval_dir is None:
        config.set('eval_dir', os.path.join(model_dir, "evaluation"), override=True)
    else:
        config.set('eval_dir', os.path.join(args.eval_dir, config.get('model_id')), override=True)
    config.set('model_dir', model_dir, override=True)  # in case the experiment folder is renamed.

    if not os.path.exists(config.get('eval_dir')):
        os.makedirs(config.get('eval_dir'))

    config.dump(config.get('eval_dir'))
    do_evaluation(config, quantitative_analysis=args.quantitative, qualitative_analysis=args.qualitative, verbose=args.verbose)