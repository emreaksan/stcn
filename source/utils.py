import time
import os
from tensorflow.python.client import timeline
import numpy as np


def get_model_dir_timestamp(prefix="", suffix="", connector="_"):
    """
    Creates a directory name based on timestamp.

    Args:
        prefix:
        suffix:
        connector: one connector character between prefix, timestamp and suffix.

    Returns:

    """
    return prefix+connector+str(int(time.time()))+connector+suffix


def create_tf_timeline(model_dir, run_metadata):
    """
    This is helpful for profiling slow Tensorflow code.

    Args:
        model_dir:
        run_metadata:

    Returns:

    """
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    timeline_file_path = os.path.join(model_dir,'timeline.json')
    with open(timeline_file_path, 'w') as f:
        f.write(ctf)


def split_data_dictionary(dictionary, split_indices, keys_frozen=[], verbose=1):
    """
    Splits the data dictionary of lists into smaller chunks. All (key,value) pairs must have the same number of
    elements. If there is an index error, then the corresponding (key, value) pair is copied directly to the new
    dictionaries.

    Args:
        dictionary (dict): data dictionary.
        split_indices (list): Each element contains a list of indices for one split. Multiple splits are supported.
        keys_frozen (list): list of keys that are to be copied directly. The remaining (key,value) pairs will be
            used in splitting. If not provided, all keys will be used.
        verbose (int): status messages.

    Returns:
        (tuple): a tuple containing chunks of new dictionaries.
    """
    # Find <key, value> pairs having an entry per sample.
    sample_level_keys = []
    dataset_level_keys = []

    num_samples = sum([len(l) for l in split_indices])
    for key, value in dictionary.items():
        if not(key in keys_frozen) and ((isinstance(value, np.ndarray) or isinstance(value, list)) and (len(value) == num_samples)):
            sample_level_keys.append(key)
        else:
            dataset_level_keys.append(key)
            print(str(key) + " is copied.")

    chunks = []
    for chunk_indices in split_indices:
        out_dict = {}

        for key in dataset_level_keys:
            out_dict[key] = dictionary[key]
        for key in sample_level_keys:
            out_dict[key] = [dictionary[key][i] for i in chunk_indices]
        chunks.append(out_dict)

    return tuple(chunks)


def get_seq_len_histogram(sequence_length_array, num_bins=10, collapse_first_and_last_bins=[1, -1]):
    """
    Creates a histogram of sequence-length.
    Args:
        sequence_length_array: numpy array of sequence length for all samples.
        num_bins:
        collapse_first_and_last_bins: selects bin edges between the provided indices by discarding from the first and
            last bins.
    Returns:
        (list): bin edges.
    """
    h, bins = np.histogram(sequence_length_array, bins=num_bins)
    if collapse_first_and_last_bins is not None:
        return [int(b) for b in bins[collapse_first_and_last_bins[0]:collapse_first_and_last_bins[1]]]
    else:
        return [int(b) for b in bins]

