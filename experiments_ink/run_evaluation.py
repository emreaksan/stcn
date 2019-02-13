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
from visualize_ink import draw_stroke_svg as visualize_ink
from configuration_ink import InkConfiguration as Configuration

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
        draw_line_plot_matplotlib(all_x_data_q_diff, all_y_data_q_diff, all_idx, all_legend_q_diff,
                                  filename + "_diff_q_mu_all")
    # Print KLD loss and stats.
    print(all_kld_loss_txt)
    with open(filename + "_kld_loss.txt", "w") as f:
        f.write(all_kld_loss_txt)


def do_evaluation(config_obj, qualitative_analysis=True, quantitative_analysis=True, pad_original=0, verbose=0):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    def postprocess(sample_):
        if config_obj.get('data_type') == C.IAMONDB:
            sample_[:, 1] = -sample_[:, 1]
        return sample_

    def visualize_samples(samples, ids, output_path, scale_factor=1, color_labels=None):
        num_samples = len(samples)
        for i in range(num_samples):
            vis_sample = samples[i]
            svg_filename = output_path + str(ids[i]) + ".svg"
            color_labels_sample = color_labels[i] if color_labels is not None else None
            visualize_ink(postprocess(vis_sample), factor=scale_factor, svg_filename=svg_filename, color_labels=color_labels_sample)

    seed = config_obj.get("seed")
    Model_cls = config_obj.model_cls
    Dataset_cls = config_obj.dataset_cls

    run_original_sample = False  # Visualize real sample
    run_reconstruction = False  # Visualize reconstruction
    run_synthesis = True  # Synthetic sample.
    run_biased_synthesis = True  # Synthetic sample based on a real one.
    run_latent_variable_plots = False
    if config_obj.get("model_type") not in [C.MODEL_STCN]:
        run_latent_variable_plots = False
    synthetic_sample_length = 2000

    if config_obj.get('data_type') == C.IAMONDB:
        sample_ids = [1, 20, 150, 1000, 1222]
        plot_sample_ids = [1000]
        factor = 10
        synthetic_sample_length = 2000
        batch_size = 32
        config_obj.config['reduce_loss'] = C.R_MEAN_SEQUENCE
    elif config_obj.get('data_type') == C.DEEPWRITING:
        sample_ids = [1, 20, 150]
        plot_sample_ids = [1]  # [1]
        factor = 0.001
        batch_size = 32
        config_obj.config['reduce_loss'] = C.R_MEAN_SEQUENCE
    else:
        raise Exception("Unknown data type.")

    # Data preprocessing configuration.
    preprocessing_ops = config_obj.get_preprocessing_ops()

    validation_dataset = Dataset_cls(config_obj.get('validation_data'), var_len_seq=True, preprocessing_ops=preprocessing_ops)
    num_validation_iterations = math.ceil(validation_dataset.num_samples/batch_size)

    # Only use 1 queue thread, otherwise validation loop it gets blocked.
    valid_data_feeder = DataFeederTF(validation_dataset, 1, batch_size, queue_capacity=1024, shuffle=False, allow_smaller_final_batch=True)
    data_placeholders = valid_data_feeder.batch_queue(dynamic_pad=validation_dataset.is_dynamic, queue_capacity=512, queue_threads=4)

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

    with tf.name_scope("validation"):
        validation_model = Model_cls(config=config_obj,
                                     session=sess,
                                     reuse=False,
                                     mode=C.EVAL,
                                     placeholders=data_placeholders,
                                     input_dims=validation_dataset.input_dims,
                                     target_dims=validation_dataset.target_dims, )
        validation_model.build_graph()

    with tf.name_scope("sampling"):
        sampling_model = Model_cls(config=config_obj,
                                   session=sess,
                                   reuse=True,
                                   mode=C.SAMPLE,
                                   placeholders=data_placeholders,
                                   input_dims=validation_dataset.input_dims,
                                   target_dims=validation_dataset.target_dims, )
        sampling_model.build_graph()

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
            _, valid_stroke_sample, valid_stroke_target = validation_dataset.fetch_sample(sample_id)
            # Prepare the sample and its reconstruction for visualization.
            if run_original_sample:
                original_sample = validation_dataset.prepare_for_visualization(valid_stroke_sample)
                out_path = os.path.join(config_obj.get('eval_dir'), "real_")
                visualize_samples(original_sample, [sample_id], out_path, scale_factor=factor)

            if run_reconstruction or run_latent_variable_plots:
                output_dict = validation_model.reconstruct(input_sequence=valid_stroke_sample, target_sequence=valid_stroke_target, use_sample_mean=True)
                if run_latent_variable_plots and sample_id in plot_sample_ids:
                    out_path = os.path.join(config_obj.get('eval_dir'), "s" + str(sample_id))
                    plots_ladder_latent_variables(output_dict["eval_dict"], out_path, plot_q_mu_diff=True, print_latent_stats=True)
                if run_reconstruction:
                    reconstructed_sample = validation_dataset.prepare_for_visualization(output_dict['sample'])
                    out_path = os.path.join(config_obj.get('eval_dir'), "reconstructed_seed" + str(seed) + "_")
                    visualize_samples(reconstructed_sample, [sample_id], out_path, scale_factor=factor)

            if run_biased_synthesis:
                if config_obj.get("model_type") in [C.MODEL_VRNN]:
                    if run_reconstruction:
                        output_dict = sampling_model.sample(seed_state=output_dict['state'], sample_length=synthetic_sample_length, use_sample_mean=True)
                    else:
                        # Fully random sampling.
                        output_dict = sampling_model.sample(seed_state=None, sample_length=synthetic_sample_length, use_sample_mean=True)
                    output_dict['sample'] = np.expand_dims(output_dict['sample'], axis=0)
                else:
                    output_dict = sampling_model.sample(seed_sequence=valid_stroke_sample, sample_length=synthetic_sample_length, use_sample_mean=True)

                # Concatenate synthetic sample with the original one.
                synthetic_sample_shape = output_dict['sample'].shape
                if pad_original > -1:
                    synthetic_sample = np.concatenate([valid_stroke_sample[:, -pad_original:], output_dict['sample']], axis=1)
                    original_sample_shape = valid_stroke_sample[:, -pad_original:].shape
                    colors = np.concatenate([np.ones((original_sample_shape[0], original_sample_shape[1])), np.ones((synthetic_sample_shape[0], synthetic_sample_shape[1]))*2], axis=1)
                else:
                    synthetic_sample = output_dict['sample']
                    colors = np.ones((synthetic_sample_shape[0], synthetic_sample_shape[1]))

                synthetic_sample = validation_dataset.prepare_for_visualization(synthetic_sample)
                out_path = os.path.join(config_obj.get('eval_dir'), "synthetic_biased_seed" + str(seed) + "_")
                visualize_samples(synthetic_sample, [sample_id], out_path, scale_factor=factor, color_labels=colors)

            if run_synthesis:
                # valid_stroke_sample[:, 0:1, :] corresponds to the initial stroke which is (0, 0, 0) for all samples.
                if config_obj.get("model_type") in [C.MODEL_VRNN]:
                    output_dict = sampling_model.sample(seed_state=None, sample_length=synthetic_sample_length, use_sample_mean=True)
                    output_dict['sample'] = np.expand_dims(output_dict['sample'], axis=0)
                else:
                    output_dict = sampling_model.sample(seed_sequence=valid_stroke_sample[:, 0:1, :], sample_length=synthetic_sample_length, use_sample_mean=True)

                synthetic_sample = output_dict['sample']

                synthetic_sample = validation_dataset.prepare_for_visualization(synthetic_sample)
                out_path = os.path.join(config_obj.get('eval_dir'), "synthetic_seed" + str(seed) + "_")
                visualize_samples(synthetic_sample, [no], out_path, scale_factor=factor)

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

    if not args.validation_data and os.path.exists(args.validation_data):
        raise Exception("Validation data not found.")

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
    do_evaluation(config, quantitative_analysis=args.quantitative, qualitative_analysis=args.qualitative, pad_original=args.pad_original, verbose=args.verbose)