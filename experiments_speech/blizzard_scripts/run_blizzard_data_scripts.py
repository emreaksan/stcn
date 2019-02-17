import os
import pickle
import tables
import numpy as np
from scipy.io import wavfile
from blizzard_data import Blizzard_tbptt

"""
Preprocessing script for Blizzard data. The same steps with VRNN (https://arxiv.org/abs/1506.02216) and
Z-forcing (https://arxiv.org/abs/1711.05411) are applied.
Download dataset from https://www.synsig.org/index.php/Blizzard_Challenge_2013.  
"""

# Path to the unsegmented Blizzard dataset. Don't add "./"
BLIZZARD_DATA_PATH = "blizzard_data_demo/unsegmented"
# Temporarily created destination for intermediate files.
TMP_DIR = "tmp"
# Destination of the dataset files.
OUTPUT_DIR = "blizzard"

os.mkdir(TMP_DIR)
os.mkdir(OUTPUT_DIR)

###
# 1-List .mp3 files
###
bash_list_mp3 = 'find . -type f -path "*/{0}/*.mp3" > {1}/list_of_files.txt'.format(BLIZZARD_DATA_PATH, TMP_DIR)
os.system(bash_list_mp3)

###
# 2-Convert mp3 into wav. Taken from
# https://github.com/jych/nips2015_vrnn/blob/275e183536a8bf4c3d30a29a1b6ccd3e8026e93c/datasets/blizzard_utils/convert_to_wav.sh
###
bash_mp3_to_wav = \
    """
    DIRNAME={0}/blizzard_wav
    if [ ! -d "$DIRNAME" ]; then
        mkdir $DIRNAME
    fi
        echo $files_list
    for f in `cat {0}/list_of_files.txt | grep .mp3 | sed -e 's/[ \t][ \t]*/ /g' | cut -d ' ' -f6`; do""".format(TMP_DIR) +\
    """
        t=`basename $f`
        ffmpeg -i "$f" -acodec pcm_s16le -ac 1 -ar 16000 $DIRNAME/${t%.mp3}.wav
    done
    """
os.system(bash_mp3_to_wav)

###
# 3-Read the wav files and dump into numpy arrays.
# https://github.com/jych/nips2015_vrnn/blob/275e183536a8bf4c3d30a29a1b6ccd3e8026e93c/datasets/blizzard_utils/make_blizzard_npy.py
###
wav_dir = os.path.join(TMP_DIR, 'blizzard_wav')
list_len = 200
file_idx = 0
data_list = []
files = [os.path.join(wav_dir, f) for f in os.listdir(wav_dir)]
for n, f in enumerate(files):
    sr, d = wavfile.read(f)
    data_list.append(d)
    if len(data_list) >= list_len:
        print("Dumping at file %i of %i" % (n, len(files)))
        pickle.dump(data_list, open(TMP_DIR + "/data_%i.npy" % file_idx, mode="wb"))
        file_idx += 1
        data_list = []
# dump the last chunk
pickle.dump(data_list, open(TMP_DIR + "/data_%i.npy" % file_idx, mode="wb"))

###
# 4-Load the numpy arrays, segment the data and calculate statistics.
# https://github.com/jych/nips2015_vrnn/blob/master/datasets/blizzard.py
###
file_name = 'blizzard_unseg_tbptt'
X_mean = None
X_std = None
train_data = Blizzard_tbptt(name='train',
                            path=TMP_DIR,
                            frame_size=200,
                            seq_len=8000,
                            file_name=file_name,
                            X_mean=X_mean,
                            X_std=X_std)
print("Num examples: " + str(train_data.num_examples()))


###
# 5-Convert to a dataset representation that is required by the STCN repository.
###
input_data_folder = TMP_DIR
hdf5_data_file = "blizzard_unseg_tbptt"
output_data_folder = OUTPUT_DIR
output_data_file = "blizzard_stcn"

# These numbers are taken from Z-forcing repository.
train_start, train_end = 0, 2040064
valid_start, valid_end = 2040064, 2152704
test_start, test_end = 2152704, 2267008 - 128

# Load raw data.
print("Reading HDF5 data file.")
hdf5_data = tables.open_file(os.path.join(input_data_folder, hdf5_data_file + ".h5"), mode='r')
dataset_all = hdf5_data.root.data

# Create training, validation and test splits.
training_dataset = dict()
validation_dataset = dict()
test_dataset = dict()

# Z-forcing paper approximates data statistics and normalizes both the inputs and targets by using the approximated mean
# and std. In order to be able to directly compare our results with them, we also use their statistics.
print("Fetching statistics.")
stats = dict(np.load(os.path.join(input_data_folder, hdf5_data_file + "_normal.npz")))
baseline_stats = dict(mean_all=np.float32(stats["X_mean"]), std_all=np.float32(stats["X_std"]))
training_dataset["statistics"] = baseline_stats
validation_dataset["statistics"] = baseline_stats
test_dataset["statistics"] = baseline_stats

print("Creating training split.")
training_dataset["samples"] = dataset_all[train_start:train_end].reshape(-1, 40, 200)
print("Creating validation split.")
validation_dataset["samples"] = dataset_all[valid_start:valid_end].reshape(-1, 40, 200)
print("Creating test split.")
test_dataset["samples"] = dataset_all[test_start:test_end].reshape(-1, 40, 200)

# Save
if training_dataset is not None:
    print("# training samples: " + str(len(training_dataset['samples'])))
    training_path = os.path.join(output_data_folder, output_data_file + "_training")
    np.savez_compressed(training_path, **training_dataset)

if validation_dataset is not None:
    print("# validation samples: " + str(len(validation_dataset['samples'])))
    validation_path = os.path.join(output_data_folder, output_data_file + "_validation")
    np.savez_compressed(validation_path, **validation_dataset)

if test_dataset is not None:
    print("# test samples: " + str(len(test_dataset['samples'])))
    validation_path = os.path.join(output_data_folder, output_data_file + "_test")
    np.savez_compressed(validation_path, **test_dataset)
