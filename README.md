# STCN: Stochastic Temporal Convolutional Networks
This repository contains the implementation of STCN: Stochastic Temporal Convolutional Networks [paper](https://openreview.net/pdf?id=HkzSQhCcK7).

The model is implemented in Python 3.5 by using [Tensorflow](https://www.tensorflow.org/install) library. 
Particularly the experiments were conducted with cuda 9, cudnn 7.2 and Tensorflow 1.10.1.
Along with the implementation, we release our pretrained models achieving the state-of-the-art performance. 
You can reproduce the numbers and results by using the evaluation scripts explained below.

## Installation
Python dependencies are listed in `pip_packages` file. You can install the packages by running
`pip install -r  pip_packages` command.

## Dataset
You should download the [Blizzard](https://www.synsig.org/index.php/Blizzard_Challenge_2013), [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1) and [IAM-OnDB](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) datasets.
We compiled the preprocessing steps applied in the previous works. 
You can find the scripts for Blizzard and TIMIT dataset in `experiments_speech` directory. We will provide the script for IAM-OnDB handwriting dataset soon. 
The [Deepwriting](https://ait.ethz.ch/projects/2019/stcn/downloads/deepwriting_dataset.tar.gz) dataset is already provided.

The instructions are provided in `run_blizzard_data_scripts.py` and `run_timit_data_scripts.py` files. 
After you download the corresponding dataset, you just need to set the `DATA_PATH` variable in the code file and run the script.

After the preprocessing steps are applied, the data is transformed into a `numpy` representation required by our training pipeline. 
It is basically a dictionary with `samples`, `targets` and data `statistics` such as mean and variance. 
It can be inspected by loading via `numpy.load`.

## Model
We use configuration files to pass model and experiment parameters. You can find the configuration files in `experiments_x/config_y` folders.
They are stored in `JSON` format. The provided configurations are the same with what we used. However, due to the randomness you may get 
better or worse results within a small margin. 

You can train a model by passing a JSON configuration file. Most entries in the configuration are self-explanatory, and the rest is described in the code (see `soruce/tf_models.py` model constructors).
You should also have a look into `source/configuration.py` file which created command-line arguments and parses JSON configuration files. 
For example the following command runs `STCN-dense` model with `GMM` outputs on `Deepwriting` dataset.
```
python run_training.py 
--experiment_name stcn_dense_gmm 
--json_file ./config_deepwriting/stcn_dense_gmm.json 
--training_data [PATH-TO]/deepwriting_v2_training.npz 
--validation_data [PATH-TO]/deepwriting_v2_validation.npz 
--save_dir [PATH-TO]/runs 
--eval_dir [PATH-TO]/evaluation_runs 
--pp_zero_mean_normalization 
--run_evaluation_after_training
```
We provide example commands for other datasets in the respective `run_training.py` file. Please check it out since the normalization flags may differ.
We apply early stopping on the validation ELBO performance. When training ends, the evaluation script is called if the `run_evaluation_after_training` flag is passed.
You can evaluate a pre-trained model by using the following command:
```
python run_evaluation.py
--seed [if not passed, training seed is used]
--quantitative 
--qualitative
--model_id [Unique timestamp ID in the experiment folder name. Something like 1549805923]
--save_dir [PATH-TO]/runs
--eval_dir [PATH-TO]/evaluation_runs
--test_data <PATH-TO>/blizzard_stcn_test.npz
```
Similarly, our models can be [downloaded](https://ait.ethz.ch/projects/2019/stcn/downloads/stcn_sota_models.tar.gz) and evaluated by just passsing the unique model ID.

If you have questions on the code, feel free to create a Github issue or contact me. 

## Citation
If you use this code or dataset in your research, please cite us as follows:
```
@inproceedings{
  aksan2018stcn,
  title={{STCN}: Stochastic Temporal Convolutional Networks},
  author={Emre Aksan and Otmar Hilliges},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://openreview.net/forum?id=HkzSQhCcK7},
}
```
