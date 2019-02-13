import numpy as np
from constants import Constants
from data_operators import Operator

C = Constants()
"""
Dataset class.

This class provides a basic interface to feed samples by using tensorflow's input pipeline (i.e., queues), hiding data
I/O latency bu using threads.

A `Dataset` object is given to `DataFeederTF` object which runs `sample_generator` method to enqueue the data queue.
The `sample_generator` method returns a generator yielding one sample at a time with shape and type specified by
`sample_shape` and `sample_tf_type`.

The way the data is passed is not restricted. A child class can read the data from numpy array, list, dictionary, etc.
"""


class BaseDataset(object):
    """
    Acts as a data container. Loads and parses data, and provides basic functionality.
    """
    def __init__(self, data_path):
        if isinstance(data_path, str):
            self.data_dict = dict(np.load(data_path))
        elif isinstance(data_path, dict):
            self.data_dict = data_path
        else:
            raise Exception("Data type isn't recognized.")

        self.num_samples = None
        self.sample_shape = None
        self.sample_np_type = None
        self.sample_tf_type = None
        self.sample_key = None

    def sample_generator(self):
        """
        Creates a generator object which returns one data sample at a time. It is used by DataFeeder objects.

        Returns:
            (generator): that yields one sample consisting of a list of data elements.
        """
        raise NotImplementedError('Method is abstract.')

    def batch_generator(self, batch_size, shuffle=True, drop_last_batch=True):
        """
        Creates a generator object which returns a batch of samples at a time.

        Returns:
            (generator): that yields a batch of samples.
        """
        raise NotImplementedError('Method is abstract.')


class Dataset(BaseDataset):
    """
    Dataset class for the most basic tasks.

    Loads data dictionary from disk. Dictionary entries with `samples` and `targets` keys are used as model inputs and
    targets, respectively. In the absence of `targets` key inputs will be used as the targets (i.e., reconstruction task).

    Data samples must be either list of samples (i.e., variable-length sequences) with shape (seq_len, feature_size) or
    numpy array of samples with shape (#_samples, seq_len, feature_size)

    Args:
        data_path: Path to the data dictionary.
        var_len_seq: If true, sequence length will be None. Otherwise, it will be calculated from data.
    """
    def __init__(self, data_path, var_len_seq=False, preprocessing_ops=None):
        super(Dataset, self).__init__(data_path)
        preprocessing_ops = preprocessing_ops or dict()

        self.samples = self.data_dict['samples']
        # self.targets = self.data_dict.get('targets', self.data_dict['samples'])
        self.targets = self.data_dict['samples']
        self.file_ids = self.data_dict.get('file_ids', None)

        self.num_samples = len(self.samples)
        assert self.num_samples == len(self.targets), "Inputs and targets have different # of samples."

        # TODO A perturbator object taking a sample of batch of samples and applies perturbation.
        self.perturbator = None
        # TODO A preprocessing object applying and undoing preprocessing routines.
        self.preprocessor = None
        # TODO A query function/object that applies a query on a sample and returns a boolean value whether to use
        # the sample or not.
        self.selector = None

        self.applied_preprocessing = self.data_dict['preprocessing'].tolist() if "preprocessing" in self.data_dict else []
        self.data_stats = self.data_dict.get('statistics').tolist() if 'statistics' in self.data_dict else {}
        self.data_stats["normalize_targets"] = True

        self.input_feature_size = self.samples[0].shape[-1]
        self.target_feature_size = self.targets[0].shape[-1]
        self.sequence_lengths = self.__extract_seq_len()

        # preprocessor is the object applying normalization.
        self.preprocessor = Operator.create(**{**preprocessing_ops, **self.data_stats})

        # Apply shifting side-effects: shifting the inputs by 1 time-step to get targets.
        if preprocessing_ops.get(C.PP_SHIFT, False):
            self.sequence_lengths = self.sequence_lengths - 1

        # Models require input and target dimensionality. `*_dims` members are useful if the inputs and targets are
        # concatenation of different modalities. They are used to split the input/target into components by the model.
        self.input_dims = [self.input_feature_size]
        self.target_dims = [self.target_feature_size]

        # The dimensions with None will be padded if seq_len isn't passed.
        self.sequence_length = None if var_len_seq else self.__get_seq_len()
        self.is_dynamic = self.sequence_length is None

        # Sequence length, input, target, idx
        self.sample_shape = [[], [self.sequence_length, sum(self.input_dims)], [self.sequence_length, sum(self.target_dims)], []]
        self.sample_np_type = [np.int32, np.float32, np.float32, np.int32]
        self.sample_key = [C.PL_SEQ_LEN, C.PL_INPUT, C.PL_TARGET, C.PL_IDX]

    def unnormalize(self, sample):
        """
        Args:
            sample:

        Returns:
        """
        sample_copy = np.copy(sample)

        is_batch = True
        if sample.ndim == 2:
            sample_copy = np.expand_dims(sample, axis=0)
            is_batch = False

        if self.preprocessor is not None:
            sample_copy, _ = self.preprocessor.undo(sample_copy)
            sample_copy = sample_copy[0] if is_batch is False else sample_copy

        return sample_copy

    def prepare_for_visualization(self, sample):
        """
        Prepare the given sample for visualization by undoing normalization and representation related operations.

        Args:
            sample:

        Returns:
        """
        return self.unnormalize(sample)

    def preprocess_sample(self, input_sample, target_sample):
        if self.preprocessor is not None:
            input_sample, target_sample = self.preprocessor.apply(np.expand_dims(input_sample, axis=0),
                                                                  np.expand_dims(target_sample, axis=0))
        return input_sample[0], target_sample[0]

    def sample_generator(self):
        """
        Creates a generator object which returns one data sample at a time. It is used by DataFeeder objects.

        Returns:
            (generator): each sample is a list of data elements.
        """
        for idx, [input_sample, target_sample, seq_len] in enumerate(zip(self.samples, self.targets, self.sequence_lengths)):
            if self.perturbator is not None:
                input_sample = self.perturbator(input_sample)

            use_sample = True
            if self.selector is not None:
                use_sample = self.selector(input_sample)

            if use_sample:
                input_sample, target_sample = self.preprocess_sample(input_sample, target_sample)
                yield [seq_len, input_sample, target_sample, idx]

    def batch_generator(self, batch_size, epoch=1, shuffle=True, drop_last_batch=True):
        """
        Creates a generator object which returns a batch of samples at a time.

        Args:
            batch_size (int): how many samples per batch to load.
            epoch (int): number of iterations over all data samples.
            shuffle (bool): set to True to have the data reshuffled at every epoch (default: True).
            drop_last_batch (bool): set to True to drop the last incomplete batch, if the dataset size is not divisible
                by the batch size (default: True).
        Returns:
            (generator):
        """
        for e in range(epoch):
            if shuffle:
                indices = np.random.permutation(self.num_samples)
            else:
                indices = np.arange(self.num_samples)

            num_samples = len(indices)
            if drop_last_batch:
                num_samples -= num_samples%batch_size

            for i in range(0, num_samples, batch_size):
                batch_sample_idx = indices[i:i + batch_size]
                batch_seq_len = self.sequence_lengths[batch_sample_idx]
                max_len = batch_seq_len.max()

                batch_inputs = np.zeros((batch_size, max_len, self.input_feature_size))
                batch_targets = np.zeros((batch_size, max_len, self.target_feature_size))
                batch_mask = np.zeros((batch_size, max_len))
                for id, sample_idx in enumerate(batch_sample_idx):
                    batch_inputs[id] = self.samples[sample_idx]
                    batch_targets[id] = self.targets[sample_idx]
                    batch_mask[id] = np.ones((batch_seq_len[id]))

                yield [batch_seq_len, batch_inputs, batch_targets, batch_mask]

    def fetch_sample(self, sample_idx, clipping_allowed=False):
        """
        Prepares one data sample (i.e. return of sample_generator) given index.
        Args:
            sample_idx:
            clipping_allowed: if multiple samples are asked and they are variable-length clip the longer ones.
        Returns:
        """
        is_list = True
        if isinstance(sample_idx, int):
            sample_idx = [sample_idx]
            is_list = False

        shortest_seq_len = np.inf
        seq_len_list, input_sample_list, target_sample_list = [], [], []
        for i in sample_idx:
            input_sample = self.samples[i]
            target_sample = self.targets[i]
            seq_len = self.sequence_lengths[i]
            shortest_seq_len = seq_len if shortest_seq_len > seq_len else shortest_seq_len

            if self.perturbator is not None:
                input_sample = self.perturbator(input_sample)

            if self.preprocessor is not None:
                input_sample, target_sample = self.preprocessor.apply(np.expand_dims(input_sample, axis=0), np.expand_dims(target_sample, axis=0))
                input_sample = input_sample[0]
                target_sample = target_sample[0]

            seq_len_list.append(np.array([[seq_len]]))
            input_sample_list.append(np.expand_dims(input_sample, axis=0))
            target_sample_list.append(np.expand_dims(target_sample, axis=0))

        if is_list:
            if clipping_allowed:
                for i in range(len(sample_idx)):
                    seq_len_list[i] = np.array([[shortest_seq_len]])
                    input_sample_list[i] = input_sample_list[i][:, 0:shortest_seq_len]
                    target_sample_list[i] = target_sample_list[i][:, 0:shortest_seq_len]
            return [np.concatenate(seq_len_list, axis=0), np.concatenate(input_sample_list, axis=0), np.concatenate(target_sample_list, axis=0)]
        else:
            return [seq_len_list[0], input_sample_list[0], target_sample_list[0]]

    def __extract_seq_len(self):
        """
        Returns (np.array):
            List of lengths of each sequence sample in the dataset.
        """
        return np.array([s.shape[0] for s in self.samples], dtype=np.int32)

    def __get_seq_len(self):
        """
        Returns (int or None):
            Sequence length of samples in the dataset. If the samples are variable-length then returns None. If dataset
            is already padded (i.e., preprocessing) then returns the fixed sample length, because padding is not
            required.
        """
        if max(self.sequence_lengths) == min(self.sequence_lengths):
            return min(self.sequence_lengths)
        else:
            return None


class PaddedDataset(Dataset):
    """
    Dataset class for the most basic tasks. The samples are expected to be padded and corresponding masks should be
    provided in data dictionary.

    Loads data dictionary from disk. Dictionary entries with `samples` and `targets` keys are used as model inputs and
    targets, respectively. In the absence of `targets` key inputs will be used as the targets (i.e., reconstruction task).

    Data samples must be either list of samples (i.e., variable-length sequences) with shape (seq_len, feature_size) or
    numpy array of samples with shape (#_samples, seq_len, feature_size).

    Args:
        data_path: Path to the data dictionary.
    """
    def __init__(self, data_path, preprocessing_ops={}):
        super(PaddedDataset, self).__init__(data_path, var_len_seq=False, preprocessing_ops=preprocessing_ops)

        self.masks = self.data_dict.get('masks', None)  # If sequences are already padded.

        self.sequence_lengths = self.__extract_seq_len()
        # The dimensions with None will be padded if seq_len isn't passed.
        self.sequence_length = self.__get_seq_len()
        self.is_dynamic = self.sequence_length is None

        if self.preprocessor is not None:
            self.sequence_lengths = self.sequence_lengths + self.preprocessor.side_effects.get(C.SE_PP_SEQ_LEN_DIFF, 0)

        assert not self.is_dynamic, "Samples must be padded."

    def batch_generator(self, batch_size, shuffle=True, drop_last_batch=True, return_mask=False):
        """
        Creates a generator object which returns one batch of samples at a time.

        Args:
            batch_size (int): how many samples per batch to load
            shuffle (bool): set to True to have the data reshuffled at every epoch (default: True).
            drop_last_batch (bool): set to True to drop the last incomplete batch, if the dataset size is not divisible
                by the batch size (default: True).
            return_mask (bool): returns binary masks instead of sequence lengths. Note that the data must be padded
                beforehand (default: False).

        Returns:
            (generator):
        """
        # Data is already padded. Create masks.
        if return_mask and self.masks is None:
            self.masks = np.zeros((self.num_samples, self.sequence_length))
            for idx, seq_len in enumerate(self.sequence_lengths):
                self.masks[idx, 0:seq_len] = np.ones(seq_len)

        def chunk(array, size):
            """
            Yields equal-sized chunks.

            Args:
                array:
                size: size of each chunk.
            """
            num_samples = len(array)
            if drop_last_batch:
                num_samples -= num_samples%size

            for i in range(0, num_samples, size):
                yield array[i:i + size]

        if shuffle:
            indices = np.random.permutation(self.num_samples)
        else:
            indices = np.arange(self.num_samples)

        for batch_indices in chunk(indices, batch_size):
            batch_input = self.samples[batch_indices]
            batch_target = self.targets[batch_indices]

            if return_mask:
                batch_seq_len = self.masks[batch_indices]
            else:
                batch_seq_len = self.sequence_lengths[batch_indices]

            yield [batch_seq_len, batch_input, batch_target]

    def __extract_seq_len(self):
        """
        Returns (np.array):
            List of lengths of each sequence sample in the dataset.
        """
        if self.masks is None:
            return np.array([s.shape[0] for s in self.samples], dtype=np.int32)
        else:
            return self.masks.sum(axis=1).astype(np.int32)

    def __get_seq_len(self):
        """
        Returns (int or None):
            Sequence length of samples in the dataset. If the samples are variable-length then returns None. If dataset
            is already padded (i.e., preprocessing) then returns the fixed sample length, because padding is not
            required.

        """
        sample_seq_lens = [sample.shape[0] for sample in self.samples]
        if max(sample_seq_lens) == min(sample_seq_lens):
            return min(sample_seq_lens)
        else:
            return None


