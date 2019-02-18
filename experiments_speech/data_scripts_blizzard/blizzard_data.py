from __future__ import division

import os
import tables
import fnmatch
import numpy as np
import scipy.signal
from multiprocessing import Process, Queue
from numpy.lib.stride_tricks import as_strided

"""
Taken from https://github.com/jych/nips2015_vrnn/
"""

def tolist(arg):
    if type(arg) is not list:
        if isinstance(arg, tuple):
            return list(arg)
        else:
            return [arg]
    return arg


def totuple(arg):
    if type(arg) is not tuple:
        if isinstance(arg, list):
            return tuple(arg)
        else:
            return (arg,)
    return arg


def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    """Generate a new array that chops the given array along the given axis
    into overlapping frames.
    This code has been implemented by Anne Archibald and has been discussed
    on the ML.
    Parameters
    ----------
    a : array-like
        The array to segment
    length : int
        The length of each frame
    overlap : int, optional
        The number of array elements by which the frames should overlap
    axis : int, optional
        The axis to operate on; if None, act on the flattened array
    end : {'cut', 'wrap', 'end'}, optional
        What to do with the last frame, if the array is not evenly
        divisible into pieces.
            - 'cut'   Simply discard the extra values
            - 'wrap'  Copy values from the beginning of the array
            - 'pad'   Pad with a constant value
    endvalue : object
        The value to use for end='pad'
    Examples
    --------
    > segment_axis(arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])
    Notes
    -----
    The array is not copied unless necessary (either because it is
    unevenly strided and being flattened or because end is set to
    'pad' or 'wrap').
    use as_strided
    """

    if axis is None:
        a = np.ravel(a) # may copy
        axis = 0

    l = a.shape[axis]

    if overlap>=length:
        raise ValueError("frames cannot overlap by more than 100%")
    if overlap<0 or length<=0:
        raise ValueError("overlap must be nonnegative and length must be positive")

    if l<length or (l-length)%(length-overlap):
        if l>length:
            roundup = length + \
                      (1+(l-length)//(length-overlap))*(length-overlap)
            rounddown = length + \
                        ((l-length)//(length-overlap))*(length-overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown<l<roundup
        assert roundup==rounddown+(length-overlap) or \
               (roundup==length and rounddown==0)
        a = a.swapaxes(-1,axis)

        if end=='cut':
            a = a[...,:rounddown]
        elif end in ['pad','wrap']: # copying will be necessary
            s = list(a.shape)
            s[-1]=roundup
            b = np.empty(s,dtype=a.dtype)
            b[...,:l] = a
            if end=='pad':
                b[...,l:] = endvalue
            elif end=='wrap':
                b[...,l:] = a[...,:roundup-l]
            a = b

        a = a.swapaxes(-1,axis)


    l = a.shape[axis]
    if l==0:
        raise ValueError("Not enough data points to segment array in 'cut' mode; try 'pad' or 'wrap'")
    assert l>=length
    assert (l-length)%(length-overlap) == 0
    n = 1+(l-length)//(length-overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis] + (n,length) + a.shape[axis+1:]
    newstrides = a.strides[:axis] + ((length-overlap)*s, s) + \
                 a.strides[axis+1:]

    try:
        return as_strided(a, strides=newstrides, shape=newshape)
    except TypeError:
        warnings.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis] + ((length-overlap)*s, s) + \
                     a.strides[axis+1:]
        return as_strided(a, strides=newstrides, shape=newshape)

def complex_to_real(X):
    """
    WRITEME
    Parameters
    ----------
    X : list of complex vectors
    Notes
    -----
    This function assumes X as 2D
    """
    new_X = []
    for i in range(len(X)):
        x = X[i]
        new_x = np.concatenate([np.real(x), np.imag(x)])
        new_X.append(new_x)
    return np.array(new_X)



class _blizzardEArray(tables.EArray):
    pass


def fetch_blizzard(data_path, shuffle=0, sz=32000, file_name="full_blizzard.h5"):

    hdf5_path = os.path.join(data_path, file_name)

    if not os.path.exists(hdf5_path):
        data_matches = []

        for root, dir_names, file_names in os.walk(data_path):
            for filename in fnmatch.filter(file_names, 'data_*.npy'):
                data_matches.append(os.path.join(root, filename))

        # sort in proper order
        data_matches = sorted(data_matches,
                              key=lambda x: int(
                                  x.split("/")[-1].split("_")[-1][0]))

        # setup tables
        compression_filter = tables.Filters(complevel=5, complib='blosc')
        hdf5_file = tables.open_file(hdf5_path, mode='w')
        data = hdf5_file.create_earray(hdf5_file.root, 'data',
                                      tables.Int16Atom(),
                                      shape=(0, sz),
                                      filters=compression_filter,)

        for n, f in enumerate(data_matches):
            print("Reading file %s" % (f))

            with open(f) as fp:
                # Array of arrays, ragged
                d = np.load(fp)

                if shuffle:
                    rnd_idx = np.random.permutation(len(d))
                    d = d[rnd_idx]

                for n, di in enumerate(d):
                    print("Processing line %i of %i" % (n+1, len(d)))

                    if len(di.shape) > 1:
                        di = di[:, 0]

                    e = [r for r in range(0, len(di), sz)]
                    e.append(None)
                    starts = e[:-1]
                    stops = e[1:]
                    endpoints = zip(starts, stops)

                    for i, j in endpoints:
                        di_new = di[i:j]

                        # zero pad
                        if len(di_new) < sz:
                            di_large = np.zeros((sz,), dtype='int16')
                            di_large[:len(di_new)] = di_new
                            di_new = di_large

                        data.append(di_new[None])

        hdf5_file.close()

    hdf5_file = tables.open_file(hdf5_path, mode='r')

    return hdf5_file.root.data


def fetch_blizzard_tbptt(data_path, sz=8000, batch_size=100, file_name="blizzard_tbptt.h5"):

    hdf5_path = os.path.join(data_path, file_name)

    if not os.path.exists(hdf5_path):
        data_matches = []

        for root, dir_names, file_names in os.walk(data_path):
            for filename in fnmatch.filter(file_names, 'data_*.npy'):
                data_matches.append(os.path.join(root, filename))

        # sort in proper order
        data_matches = sorted(data_matches,
                              key=lambda x: int(
                                  x.split("/")[-1].split("_")[-1][0]))

        # setup tables
        compression_filter = tables.Filters(complevel=5, complib='blosc')
        hdf5_file = tables.open_file(hdf5_path, mode='w')
        data = hdf5_file.create_earray(hdf5_file.root, 'data',
                                      tables.Int16Atom(),
                                      shape=(0, sz),
                                      filters=compression_filter,)

        for n, f in enumerate(data_matches):
            print("Reading file %s" % (f))

            # with open(f) as fp:
            # Array of arrays, ragged
            #    d = np.load(fp)
            d = np.load(f)
            large_d = d[0]

            for i in range(1, len(d)):
                print("Processing line %i of %i" % (i+1, len(d)))
                di = d[i]

                if len(di.shape) > 1:
                    di = di[:, 0]

                large_d = np.concatenate([large_d, di])

            chunk_size = int(np.float(len(large_d) / batch_size))
            seg_d = segment_axis(large_d, chunk_size, 0)
            num_batch = int(np.float((seg_d.shape[-1] - 1)/float(sz)))

            for i in range(num_batch):
                batch = seg_d[:, i*sz:(i+1)*sz]

                for j in range(batch_size):
                    data.append(batch[j][None])

        hdf5_file.close()

    hdf5_file = tables.open_file(hdf5_path, mode='r')

    return hdf5_file.root.data


class SequentialPrepMixin(object):
    """
    Preprocessing mixin for sequential data
    """
    def norm_normalize(self, X, avr_norm=None):
        """
        Unify the norm of each sequence in X
        Parameters
        ----------
        X       : list of lists or ndArrays
        avr_nom : Scalar
        """
        if avr_norm is None:
            avr_norm = 0
            for i in range(len(X)):
                euclidean_norm = np.sqrt(np.square(X[i].sum()))
                X[i] /= euclidean_norm
                avr_norm += euclidean_norm
            avr_norm /= len(X)
        else:
            X = [x[i] / avr_norm for x in X]
        return X, avr_norm

    def global_normalize(self, X, X_mean=None, X_std=None):
        """
        Globally normalize X into zero mean and unit variance
        Parameters
        ----------
        X      : list of lists or ndArrays
        X_mean : Scalar
        X_std  : Scalar
        Notes
        -----
        Compute varaince using the relation
        >>> Var(X) = E[X^2] - E[X]^2
        """
        if X_mean is None or X_std is None:
            X_len = np.array([len(x) for x in X]).sum()
            X_mean = np.array([x.sum() for x in X]).sum() / X_len
            X_sqr = np.array([(x**2).sum() for x in X]).sum() / X_len
            X_std = np.sqrt(X_sqr - X_mean**2)
            X = (X - X_mean) / X_std
        else:
            X = (X - X_mean) / X_std
        return (X, X_mean, X_std)

    def standardize(self, X, X_max=None, X_min=None):
        """
        Standardize X such that X \in [0, 1]
        Parameters
        ----------
        X     : list of lists or ndArrays
        X_max : Scalar
        X_min : Scalar
        """
        if X_max is None or X_min is None:
            X_max = np.array([x.max() for x in X]).max()
            X_min = np.array([x.min() for x in X]).min()
            X = (X - X_min) / (X_max - X_min)
        else:
            X = (X - X_min) / (X_max - X_min)
        return (X, X_max, X_min)

    def numpy_rfft(self, X):
        """
        Apply real FFT to X (numpy)
        Parameters
        ----------
        X     : list of lists or ndArrays
        """
        X = np.array([np.fft.rfft(x) for x in X])
        return X

    def numpy_irfft(self, X):
        """
        Apply real inverse FFT to X (numpy)
        Parameters
        ----------
        X     : list of lists or ndArrays
        """
        X = np.array([np.fft.irfft(x) for x in X])
        return X

    def rfft(self, X):
        """
        Apply real FFT to X (scipy)
        Parameters
        ----------
        X     : list of lists or ndArrays
        """
        X = np.array([scipy.fftpack.rfft(x) for x in X])
        return X

    def irfft(self, X):
        """
        Apply real inverse FFT to X (scipy)
        Parameters
        ----------
        X     : list of lists or ndArrays
        """
        X = np.array([scipy.fftpack.irfft(x) for x in X])
        return X

    def stft(self, X):
        """
        Apply short-time Fourier transform to X
        Parameters
        ----------
        X     : list of lists or ndArrays
        """
        X = np.array([scipy.fft(x) for x in X])
        return X

    def istft(self, X):
        """
        Apply short-time Fourier transform to X
        Parameters
        ----------
        X     : list of lists or ndArrays
        """
        X = np.array([scipy.real(scipy.ifft(x)) for x in X])
        return X

    def fill_zero1D(self, x, pad_len=0, mode='righthand'):
        """
        Given variable lengths sequences,
        pad zeros w.r.t to the maximum
        length sequences and create a
        dense design matrix
        Parameters
        ----------
        X       : list or 1D ndArray
        pad_len : integer
            if 0, we consider that output should be
            a design matrix.
        mode    : string
            Strategy to fill-in the zeros
            'righthand': pad the zeros at the right space
            'lefthand' : pad the zeros at the left space
            'random'   : pad the zeros with randomly
                         chosen left space and right space
        """
        if mode == 'lefthand':
            new_x = np.concatenate([np.zeros((pad_len)), x])
        elif mode == 'righthand':
            new_x = np.concatenate([x, np.zeros((pad_len))])
        elif mode == 'random':
            new_x = np.concatenate(
                [np.zeros((pad_len)), x, np.zeros((pad_len))]
            )
        return new_x

    def fill_zero(self, X, pad_len=0, mode='righthand'):
        """
        Given variable lengths sequences,
        pad zeros w.r.t to the maximum
        length sequences and create a
        dense design matrix
        Parameters
        ----------
        X       : list of ndArrays or lists
        pad_len : integer
            if 0, we consider that output should be
            a design matrix.
        mode    : string
            Strategy to fill-in the zeros
            'righthand': pad the zeros at the right space
            'lefthand' : pad the zeros at the left space
            'random'   : pad the zeros with randomly
                         chosen left space and right space
        """
        if pad_len == 0:
            X_max = np.array([len(x) for x in X]).max()
            new_X = np.zeros((len(X), X_max))
            for i, x in enumerate(X):
                free_ = X_max - len(x)
                if mode == 'lefthand':
                    new_x = np.concatenate([np.zeros((free_)), x], axis=1)
                elif mode == 'righthand':
                    new_x = np.concatenate([x, np.zeros((free_))], axis=1)
                elif mode == 'random':
                    j = np.random.randint(free_)
                    new_x = np.concatenate(
                        [np.zeros((j)), x, np.zeros((free_ - j))],
                        axis=1
                    )
                new_X[i] = new_x
        else:
            new_X = []
            for x in X:
                if mode == 'lefthand':
                    new_x = np.concatenate([np.zeros((pad_len)), x], axis=1)
                elif mode == 'righthand':
                    new_x = np.concatenate([x, np.zeros((pad_len))], axis=1)
                elif mode == 'random':
                    new_x = np.concatenate(
                        [np.zeros((pad_len)), x, np.zeros((pad_len))],
                         axis=1
                    )
                new_X.append(new_x)
        return new_X

    def reverse(self, X):
        """
        Reverse each sequence of X
        Parameters
        ----------
        X       : list of ndArrays or lists
        """
        new_X = []
        for x in X:
            new_X.append(x[::-1])
        return new_X


class Data(object):
    """
    Abstract class for data
    Parameters
    ----------
    .. todo::
    """
    def __init__(self, name=None, path=None, multi_process=0):
        self.name = name
        self.data = self.load(path)
        self.multi_process = multi_process
        if multi_process > 0:
            self.queue = Queue(2**15)
            processes = [None] * multi_process
            for mid in range(multi_process):
                processes[mid] = Process(target=self.multi_process_slices,
                                         args=(mid,))
                processes[mid].start()

    def multi_process_slices(self, mid=-1):
        raise NotImplementedError(
            str(type(self)) + " does not implement Data.multi_process_slices.")

    def load(self, path):
        return np.load(path)

    def slices(self):
        raise NotImplementedError(
            str(type(self)) + " does not implement Data.slices.")

    def num_examples(self):
        return max(mat.shape[0] for mat in self.data)


class TemporalSeries(Data):
    """
    Abstract class for temporal data.
    We use TemporalSeries when the data contains variable length
    seuences, otherwise, we use DesignMatrix.
    Parameters
    ----------
    .. todo::
    """
    def slices(self, start, end):
        return (mat[start:end].swapaxes(0, 1)
                for mat in self.data)

    def create_mask(self, batch):
        samples_len = [len(sample) for sample in batch]
        max_sample_len = max(samples_len)
        mask = np.zeros((max_sample_len, len(batch)), dtype=batch[0].dtype)
        for i, sample_len in enumerate(samples_len):
            mask[:sample_len, i] = 1.
        return mask

    def zero_pad(self, batch):
        max_sample_len = max(len(sample) for sample in batch)
        rval = np.zeros((len(batch), max_sample_len, batch[0].shape[-1]),
                        dtype=batch[0].dtype)
        for i, sample in enumerate(batch):
            rval[i, :len(sample)] = sample
        return rval.swapaxes(0, 1)

    def create_mask_and_zero_pad(self, batch):
        samples_len = [len(sample) for sample in batch]
        max_sample_len = max(samples_len)
        mask = np.zeros((max_sample_len, len(batch)), dtype=batch[0].dtype)
        if batch[0].ndim == 1:
            rval = np.zeros((max_sample_len, len(batch)), dtype=batch[0].dtype)
        else:
            rval = np.zeros((max_sample_len, len(batch), batch[0].shape[1]),
                            dtype=batch[0].dtype)
        for i, (sample, sample_len) in enumerate(zip(batch, samples_len)):
            mask[:sample_len, i] = 1.
            if batch[0].ndim == 1:
                rval[:sample_len, i] = sample
            else:
                rval[:sample_len, i, :] = sample
        return rval, mask


class Blizzard(TemporalSeries, SequentialPrepMixin):
    """
    Blizzard dataset batch provider
    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 X_mean=None,
                 X_std=None,
                 shuffle=0,
                 seq_len=32000,
                 use_window=0,
                 use_spec=0,
                 frame_size=200,
                 overlap=0,
                 file_name="full_blizzard",
                 **kwargs):

        self.X_mean = X_mean
        self.X_std = X_std
        self.shuffle = shuffle
        self.seq_len = seq_len
        self.use_window = use_window
        self.use_spec = use_spec
        self.frame_size = frame_size
        self.file_name = file_name
        self.overlap = overlap

        if self.use_window or self.use_spec:
            if self.use_spec:
                if not is_power2(self.frame_size):
                    raise ValueError("Provide a number which is power of 2,\
                                      for fast speed of DFT.")

            if np.mod(self.frame_size, 2) == 0:
                self.overlap = self.frame_size / 2
            else:
                self.overlap = (self.frame_size - 1) / 2

            self.window = np.maximum(scipy.signal.hann(self.frame_size)[None, :], 1e-4).astype(np.float32)

        super(Blizzard, self).__init__(**kwargs)

    def load(self, data_path):

        X = fetch_blizzard(data_path, self.shuffle, self.seq_len, self.file_name+'.h5')

        if (self.X_mean is None or self.X_std is None) and not self.use_spec:
            prev_mean = None
            prev_var = None
            n_seen = 0
            n_inter = 10000
            range_end = np.int(np.ceil(len(X) / float(n_inter)))

            for i in range(range_end):
                n_seen += 1
                i_start = i*n_inter
                i_end = min((i+1)*n_inter, len(X))

                if prev_mean is None:
                    prev_mean = X[i_start:i_end].mean()
                    prev_var = 0.
                else:
                    curr_mean = prev_mean +\
                        (X[i_start:i_end] - prev_mean).mean() / n_seen
                    curr_var = prev_var +\
                        ((X[i_start:i_end] - prev_mean) *
                         (X[i_start:i_end] - curr_mean)).mean()
                    prev_mean = curr_mean
                    prev_var = curr_var

                print("[%d / %d]" % (i+1, range_end))

            save_file_name = self.file_name + '_normal.npz'
            self.X_mean = prev_mean
            self.X_std = np.sqrt(prev_var / n_seen)
            np.savez(data_path + save_file_name, X_mean=self.X_mean, X_std=self.X_std)

        return X
    """
    def theano_vars(self):
        return T.tensor3('x', dtype=theano.config.floatX)

    def test_theano_vars(self):
        return T.matrix('x', dtype=theano.config.floatX)
    """
    def slices(self, start, end):

        batch = np.array(self.data[start:end], dtype=np.float32)

        if self.use_spec:
            batch = self.apply_fft(batch)
            batch = self.log_magnitude(batch)
            batch = self.concatenate(batch)
        else:
            batch -= self.X_mean
            batch /= self.X_std
            if self.use_window:
                batch = self.apply_window(batch)
            else:
                batch = np.asarray([segment_axis(x, self.frame_size, 0) for x in batch])

        batch = batch.transpose(1, 0, 2)

        return totuple(batch)

    def apply_window(self, batch):

        batch = np.array([self.window * segment_axis(x, self.frame_size,
                                                     self.overlap, end='pad')
                          for x in batch])

        return batch

    def apply_fft(self, batch):

        batch = np.array([self.numpy_rfft(self.window *
                                          segment_axis(x, self.frame_size,
                                                       self.overlap, end='pad'))
                          for x in batch])

        return batch

    def apply_ifft(self, batch):

        batch = np.array([self.numpy_irfft(example) for example in batch])

        return batch

    def log_magnitude(self, batch):

        batch_shape = batch.shape
        batch_reshaped = batch.reshape((batch_shape[0] *
                                        batch_shape[1],
                                        batch_shape[2]))

        # Transform into polar domain (magnitude & phase)
        mag, phase = R2P(batch_reshaped)
        log_mag = np.log10(mag + 1.)

        # Transform back into complex domain (real & imag)
        batch_normalized = P2R(log_mag, phase)

        #batch_normalized = batch_reshaped * log_mag / mag
        new_batch = batch_normalized.reshape((batch_shape[0],
                                              batch_shape[1],
                                              batch_shape[2]))

        return new_batch

    def pow_magnitude(self, batch):

        batch_shape = batch.shape
        batch_reshaped = batch.reshape((batch_shape[0] *
                                        batch_shape[1],
                                        batch_shape[2]))

        # Transform into polar domain (magnitude & phase)
        log_mag, phase = R2P(batch_reshaped)
        mag = 10**log_mag - 1.

        # Transform back into complex domain (real & imag)
        batch_unnormalized = P2R(mag, phase)

        #batch_unnormalized = batch_reshaped * mag / log_mag
        new_batch = batch_unnormalized.reshape((batch_shape[0],
                                                batch_shape[1],
                                                batch_shape[2]))

        return new_batch

    def concatenate(self, batch):

        batch_shape = batch.shape
        batch_reshaped = batch.reshape((batch_shape[0] *
                                        batch_shape[1],
                                        batch_shape[2]))
        batch_concatenated = complex_to_real(batch_reshaped)
        new_batch = batch_concatenated.reshape((batch_shape[0],
                                                batch_shape[1],
                                                batch_concatenated.shape[-1]))
        new_batch = new_batch.astype(np.float32)

        return new_batch


class Blizzard_tbptt(Blizzard):
    """
    Blizzard dataset batch provider
    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 batch_size=100,
                 file_name='blizzard_tbptt',
                 range_start=0,
                 range_end=None,
                 **kwargs):

        self.batch_size = batch_size
        self.range_start = range_start
        self.range_end = range_end
        super(Blizzard_tbptt, self).__init__(file_name=file_name, **kwargs)

    def load(self, data_path):

        X = fetch_blizzard_tbptt(data_path, self.seq_len, self.batch_size,
                                 file_name=self.file_name+'.h5')

        if (self.X_mean is None or self.X_std is None) and not self.use_spec:
            prev_mean = None
            prev_var = None
            n_seen = 0
            n_inter = 10000  # Lower values provide more precise approximation.
            range_start = self.range_start

            if self.range_end is not None:
                range_end = np.int(np.ceil(self.range_end / float(n_inter)))
            else:
                range_end = np.int(np.ceil(len(X) / float(n_inter)))

            for i in range(range_start, range_end):
                n_seen += 1
                i_start = i*n_inter
                i_end = min((i+1)*n_inter, len(X))

                if prev_mean is None:
                    prev_mean = X[i_start:i_end].mean()
                    prev_var = 0.
                else:
                    curr_mean = prev_mean + (X[i_start:i_end] - prev_mean).mean() / n_seen
                    curr_var = prev_var +\
                        ((X[i_start:i_end] - prev_mean) *\
                         (X[i_start:i_end] - curr_mean)).mean()
                    prev_mean = curr_mean
                    prev_var = curr_var

                print("[%d / %d]" % (i+1, range_end))

            save_file_name = self.file_name + '_normal.npz'
            self.X_mean = prev_mean
            self.X_std = np.sqrt(prev_var / n_seen)
            print("mean: " + str(self.X_mean))
            print("std: " + str(self.X_std))
            np.savez(os.path.join(data_path, save_file_name), X_mean=self.X_mean, X_std=self.X_std)

        return X


def P2R(magnitude, phase):
    return magnitude * np.exp(1j*phase)


def R2P(x):
    return np.abs(x), np.angle(x)


def is_power2(num):
    """
    States if a number is a power of two (Author: A.Polino)
    """
    return num != 0 and ((num & (num - 1)) == 0)

