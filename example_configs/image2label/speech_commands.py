# pylint: skip-file
from open_seq2seq.models import Image2Label
from open_seq2seq.encoders import ResNetEncoder
from open_seq2seq.decoders import FullyConnectedDecoder
from open_seq2seq.losses import CrossEntropyLoss
from open_seq2seq.data import SpeechCommandsDataLayer
from open_seq2seq.optimizers.lr_policies import piecewise_constant, poly_decay
import tensorflow as tf


base_model = Image2Label

dataset_version = "v1-30"
dataset_location = "data/speech_commands_v0.01"

if dataset_version == "v1-12":
  num_labels = 12
elif dataset_version == "v1-30":
  num_labels = 30
else: 
  num_labels = 35
  dataset_location = "data/speech_commands_v0.02"

base_params = {
  "random_seed": 0,
  "use_horovod": False,
  "num_gpus": 1,

  "max_steps": 10000,
  "batch_size_per_gpu": 32,
  "dtype": tf.float32,

  # "dtype": "mixed",
  # "loss_scaling": 512.0,
  # "larc_params": {
  #   "larc_eta": 0.001,
  # },

  "save_summaries_steps": 1000,
  "print_loss_steps": 10,
  "print_samples_steps": 10000,
  "eval_steps": 200,
  "save_checkpoint_steps": 10000,
  "logdir": "experiments/speech_commands",

  "optimizer": "Momentum",
  "optimizer_params": {
    "momentum": 0.90, # 0.95
  },
  "lr_policy": poly_decay,
  "lr_policy_params": {
    "learning_rate": 0.1,
    "power": 2,
  },
  
  # "lr_policy": piecewise_constant,
  # "lr_policy_params": {
  #   "learning_rate": 0.1,
  #   "boundaries": [1000, 2000, 3000, 4000],
  #   "decay_rates": [0.1, 0.01, 0.001, 1e-4],
  # },

  "initializer": tf.variance_scaling_initializer,

  "regularizer": tf.contrib.layers.l2_regularizer,
  "regularizer_params": {
    'scale': 0.0001,
  },
  "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                'variable_norm', 'gradient_norm', 'global_gradient_norm'],
  "encoder": ResNetEncoder,
  "encoder_params": {
    'resnet_size': 50,
    "regularize_bn": False,
  },
  "decoder": FullyConnectedDecoder,
  "decoder_params": {
    "output_dim": num_labels,
  },
  "loss": CrossEntropyLoss,
  "data_layer": SpeechCommandsDataLayer,
  "data_layer_params": {
    "dataset_location": dataset_location,
    "num_audio_features": 80,
    "num_labels": num_labels,
    "cache_data": True
  },
}

train_params = {
  "data_layer_params": {
    "dataset_files": [
      "training_list.txt"
    ],
    "shuffle": True,
    "repeat": True
  },
}

eval_params = {
  "batch_size_per_gpu": 16,
  "data_layer_params": {
    "dataset_files": [
      "testing_list_labeled.txt"
    ],
    "shuffle": False,
    "repeat": False
  },
}