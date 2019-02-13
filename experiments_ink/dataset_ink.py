import numpy as np
from constants import Constants as C
from dataset import Dataset


"""
Dataset class for ink project.

This class provides a basic interface to feed samples by using tensorflow's input pipeline (i.e., queues), hiding data
I/O latency bu using threads.

A `Dataset` object is given to `DataFeederTF` object which runs `sample_generator` method to enqueue the data queue.
The `sample_generator` method returns a generator yielding one sample at a time with shape and type specified by
`sample_shape` and `sample_tf_type`.

The way the data is passed is not restricted. A child class can read the data from numpy array, list, dictionary, etc.
"""


class InkDataset(Dataset):
    """
    Dataset class for the Iamondb dataset. A handwriting stroke consists of binary pen-event and <x,y> pen position.

    Loads data dictionary from disk. Dictionary entries with `samples` and `targets` keys are used as model inputs and
    targets, respectively. In the absence of `targets` key inputs will be used as the targets (i.e., reconstruction task).

    Data samples must be either list of samples (i.e., variable-length sequences) with shape (seq_len, feature_size) or
    numpy array of samples with shape (#_samples, seq_len, feature_size).

    Args:
        data_path: Path to the data dictionary.
        var_len_seq: If true, sequence length will be None. Otherwise, it will be calculated from data.
    """
    def __init__(self, data_path, var_len_seq=False, preprocessing_ops={}):
        super(InkDataset, self).__init__(data_path, var_len_seq=var_len_seq, preprocessing_ops=preprocessing_ops)

        # Split the targets into two: (1) screen coordinates (u,v), (2) binary pen event.
        self.input_dims = [self.input_feature_size]
        self.target_dims = [2, 1]

        # Sequence length, input, target, idx
        self.sample_shape = [[], [self.sequence_length, sum(self.input_dims)], [self.sequence_length, sum(self.target_dims)], []]
        self.sample_np_type = [np.int32, np.float32, np.float32, np.int32]

        self.relative_representation = 'relative_representation' in self.applied_preprocessing
        self.offset_removal = 'origin_translation' in self.applied_preprocessing
        self.scale = 'scale' in self.data_dict['preprocessing']

    def unnormalize(self, sample):
        """
        Args:
            sample:

        Returns:
        """
        sample_copy = np.copy(sample)
        is_batch = True
        if sample.ndim == 2:
            sample_copy = np.expand_dims(sample_copy, axis=0)
            is_batch = False

        if self.preprocessor is not None:
            sample_copy, _ = self.preprocessor.undo(sample_copy)

        sample_copy[:, :, -1] = sample[:, :, -1]
        if is_batch is False:
            sample_copy = sample_copy[0] if is_batch is False else sample_copy

        return sample_copy

    def prepare_for_visualization(self, sample):
        """
        Prepare the given sample for visualization by undoing normalization and representation related operations.
        Args:
            sample: one or multiple samples.
        Returns:
        """
        return self.unnormalize(sample)
