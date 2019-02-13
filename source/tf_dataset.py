import tensorflow as tf
from dataset import Dataset, PaddedDataset

"""
To decouple tensorflow routines from standard python routines so that dataset class can still be used with other
frameworks.
"""


class DatasetTF(Dataset):
    """
    Tensorflow extension of Dataset class.
    """
    def __init__(self, data_path, var_len_seq=False, preprocessing_ops={}):
        super(DatasetTF, self).__init__(data_path, var_len_seq=var_len_seq, preprocessing_ops=preprocessing_ops)
        # Add tensorflow data types.
        self.sample_tf_type = [tf.int32, tf.float32, tf.float32, tf.int32]


class PaddedDatasetTF(PaddedDataset):
    """
    Tensorflow extension of Dataset class.
    """
    def __init__(self, data_path, preprocessing_ops={}):
        super(PaddedDatasetTF, self).__init__(data_path, preprocessing_ops=preprocessing_ops)
        # Add tensorflow data types.
        self.sample_tf_type = [tf.int32, tf.float32, tf.float32, tf.int32]