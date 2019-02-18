"""
Preprocessing script for TIMIT data. The same steps with VRNN (https://arxiv.org/abs/1506.02216),
Z-forcing (https://arxiv.org/abs/1711.05411) and SRNN (https://arxiv.org/abs/1605.07571) are applied.

You can download TIMIT dataset from https://catalog.ldc.upenn.edu/LDC93S1
"""

import os
import numpy as np
from timit_for_srnn import load_wav_files_relative_path, create_timit_samples

# Path to the TIMIT dataset.
TIMIT_DATA_PATH = '<>/TIMIT/'
OUTPUT_DIR = "data_timit"
OUTPUT_FILE_NAME = "timit_stcn"

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


def cancel_normalization(data_dicts):
    """
    Reverts the normalization.
    """

    def operate_single_data(data_dict):
        new_data_dict = dict()
        new_data_dict['samples'] = []
        new_data_dict['targets'] = []
        new_data_dict['preprocessing'] = []

        mask_key = "mask_test" if "mask_test" in data_dict else "masks"
        mask_exist = mask_key in data_dict

        mean, std = data_dict['mean'], data_dict['std']
        for idx in range(data_dict['samples'].shape[0]):
            sample = data_dict['samples'][idx]
            target = data_dict['targets'][idx]
            if mask_exist:
                seq_len = int(data_dict[mask_key][idx].sum())
                sample = sample[:seq_len]
                target = target[:seq_len]
            new_sample = sample*std + mean
            new_data_dict['samples'].append(new_sample)

            new_target = target*std + mean
            new_data_dict['targets'].append(new_target)

        return new_data_dict

    outputs = []
    for data_dict in data_dicts:
        if data_dict is not None:
            outputs.append(operate_single_data(data_dict))
        else:
            outputs.append(None)

    return outputs


def calculate_statistics(training_dict, evaluation_dicts=None, keep_dims=None):
    """
    Calculates min, max, mean and std statistics on all dimensions and feature dimension.
    """
    statistics = {}

    if keep_dims is not None:
        training_dict["samples"] = training_dict["samples"][:, :, keep_dims]
        for eval_data_dict in evaluation_dicts:
            if eval_data_dict is not None:
                eval_data_dict["samples"] = eval_data_dict["samples"][:, :, keep_dims]

    all_samples = np.vstack(training_dict['samples'])

    std_channel = all_samples.std(axis=0)
    std_channel[np.where(std_channel < 1e-6)] = 1.0
    mean_channel = all_samples.mean(axis=0)

    statistics['mean_all'] = all_samples.mean()
    statistics['std_all'] = all_samples.std()
    statistics['min_all'] = all_samples.min()
    statistics['max_all'] = all_samples.max()
    statistics['mean_channel'] = mean_channel
    statistics['std_channel'] = std_channel
    statistics['min_channel'] = all_samples.min(axis=0)
    statistics['max_channel'] = all_samples.max(axis=0)

    training_dict['statistics'] = statistics
    for eval_dict in evaluation_dicts:
        if eval_dict is not None:
            eval_dict['statistics'] = statistics

    return training_dict, evaluation_dicts

###
# 1-Run SRNN (or VRNN) preprocessing steps.
# Original timit_for_srnn.py code is in Python2. os.walk operation returns the file names in arbitrary order. Hence,
# the dataset split is not exactly the same in Python3. The timit_*_files.txt are created by first running the Python2
# script. Here we only load the file names and create the splits accordingly.
###
f_obj = open(os.path.join('timit_train_files.txt'), 'r')
train_files = f_obj.read().splitlines()

f_obj = open(os.path.join('timit_valid_files.txt'), 'r')
valid_files = f_obj.read().splitlines()

f_obj = open(os.path.join('timit_test_files.txt'), 'r')
test_files = f_obj.read().splitlines()

print("NUMBER OF TRAIN FILES", len(train_files))
print("NUMBER OF VALID FILES", len(valid_files))
print("NUMBER OF TEST FILES", len(test_files))

train_vector = load_wav_files_relative_path(TIMIT_DATA_PATH, train_files)
valid_vector = load_wav_files_relative_path(TIMIT_DATA_PATH, valid_files)
test_vector_lst = load_wav_files_relative_path(TIMIT_DATA_PATH, test_files)

u_train_vector, u_valid_vector, u_test_vector, x_train_vector, x_valid_vector, x_test_vector, mask_test, mean, std = create_timit_samples(train_vector, valid_vector, test_vector_lst)

###
# 2- Transform the data into STCN dataset representation.
###
training_dataset = dict(targets=x_train_vector, samples=u_train_vector, mean=mean, std=std, preprocessing=np.array(['normalization']))
validation_dataset = dict(targets=x_valid_vector, samples=u_valid_vector, mean=mean, std=std, preprocessing=np.array(['normalization']))
test_dataset = dict(targets=x_test_vector, samples=u_test_vector, mean=mean, std=std, masks=mask_test, preprocessing=np.array(['normalization']))

training_dataset, validation_dataset, test_dataset = cancel_normalization([training_dataset, validation_dataset, test_dataset])

training_dataset, eval_dataset = calculate_statistics(training_dataset, [validation_dataset, test_dataset])
validation_dataset, test_dataset = eval_dataset

# Save
if training_dataset is not None:
    print("# training samples: " + str(len(training_dataset['samples'])))
    training_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME + "_training")
    np.savez_compressed(training_path, **training_dataset)

if validation_dataset is not None:
    print("# validation samples: " + str(len(validation_dataset['samples'])))
    validation_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME + "_validation")
    np.savez_compressed(validation_path, **validation_dataset)

if test_dataset is not None:
    print("# test samples: " + str(len(test_dataset['samples'])))
    test_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME + "_test")
    np.savez_compressed(test_path, **test_dataset)