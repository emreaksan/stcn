import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from constants import Constants

C = Constants()


class Operator(object):
    def __init__(self, operator_obj=None):
        self.operator_obj = operator_obj

    def apply(self, input_data, target_data=None):
        """
        Applies a preprocessing operation on given input and target samples (if not None).

        Args:
            input_data (np.ndarray): shape of (batch_size, sequence_length, feature_size)
            target_data (np.ndarray): shape of (batch_size, sequence_length, feature_size)

        Returns:
        """
        return input_data.copy(), target_data.copy()

    def undo(self, input_data, target_data=None):
        """
        Undo the preprocessing operation if it is stateless. Otherwise, implements identity function.

        Args:
            input_data (np.ndarray): shape of (batch_size, sequence_length, feature_size)
            target_data (np.ndarray): shape of (batch_size, sequence_length, feature_size)

        Returns:
        """
        return input_data, target_data

    @staticmethod
    def create(**kwargs):
        operator_obj = Operator()

        if kwargs.get(C.PP_SHIFT, False):
            operator_obj = Shift(operator_obj=operator_obj)

        if kwargs.get(C.PP_ZERO_MEAN_NORM, False):
            operator_obj = NormalizeZeroMeanUnitVariance(data_mean=kwargs['mean_channel'], data_std=kwargs['std_channel'], apply_on_targets=kwargs.get('normalize_targets', True), operator_obj=operator_obj)

        if kwargs.get(C.PP_ZERO_MEAN_NORM_SEQ, False):
            operator_obj = NormalizeZeroMeanUnitVariance(data_mean=kwargs['mean_sequence'], data_std=kwargs['std_sequence'], apply_on_targets=kwargs.get('normalize_targets', True), operator_obj=operator_obj)

        if kwargs.get(C.PP_ZERO_MEAN_NORM_ALL, False):
            operator_obj = NormalizeZeroMeanUnitVariance(data_mean=kwargs['mean_all'], data_std=kwargs['std_all'], apply_on_targets=kwargs.get('normalize_targets', True), operator_obj=operator_obj)
        return operator_obj


class Shift(Operator):
    def __init__(self, shift_steps=1, operator_obj=None):
        super(Shift, self).__init__(operator_obj)
        self.shift_steps = shift_steps

    def apply(self, input_data, target_data=None):
        input_operated, target_operated = self.operator_obj.apply(input_data, target_data)

        input_operated = input_operated[:, :-self.shift_steps]
        if target_operated is not None:
            target_operated = target_operated[:, self.shift_steps:]

        return input_operated, target_operated

    def undo(self, input_data, target_data=None):
        """
        Identity function.
        """
        input_reverted, target_reverted = self.operator_obj.undo(input_data, target_data)

        return input_reverted, target_reverted


class NormalizeZeroMeanUnitVariance(Operator):
    def __init__(self, data_mean, data_std, apply_on_targets=True, operator_obj=None):
        super(NormalizeZeroMeanUnitVariance, self).__init__(operator_obj)
        self.data_mean = data_mean
        self.data_std = data_std
        self.apply_on_targets = apply_on_targets

    def apply(self, input_data, target_data=None):
        input_operated, target_operated = self.operator_obj.apply(input_data, target_data)

        input_operated = (input_operated - self.data_mean) / self.data_std
        if self.apply_on_targets and target_operated is not None:
            target_operated = (target_operated - self.data_mean) / self.data_std

        return input_operated, target_operated

    def undo(self, input_data, target_data=None):
        input_reverted = input_data*self.data_std + self.data_mean
        if self.apply_on_targets and target_data is not None:
            target_reverted = target_data*self.data_std + self.data_mean
        else:
            target_reverted = target_data

        input_reverted, target_reverted = self.operator_obj.undo(input_reverted, target_reverted)

        return input_reverted, target_reverted
