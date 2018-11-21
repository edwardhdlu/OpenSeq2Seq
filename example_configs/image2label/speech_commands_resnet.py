# pylint: skip-file
from open_seq2seq.models import Image2Label
from open_seq2seq.encoders import ResNetEncoder
from open_seq2seq.decoders import FullyConnectedDecoder
from open_seq2seq.losses import CrossEntropyLoss
from open_seq2seq.data import SpeechCommandsDataLayer
from open_seq2seq.optimizers.lr_policies import piecewise_constant
import tensorflow as tf


base_model = Image2Label

base_params = {
  "random_seed": 0,
  "use_horovod": False,
  "num_epochs": 100,

  "num_gpus": 1,
  "batch_size_per_gpu": 32,
  "dtype": tf.float32,

  "save_summaries_steps": 2000,
  "print_loss_steps": 100,
  "print_samples_steps": 2000,
  "eval_steps": 5000,
  "save_checkpoint_steps": 5000,
  "logdir": "experiments/speech_commands_resnet",

  "optimizer": "Momentum",
  "optimizer_params": {
    "momentum": 0.90,
  },
  "lr_policy": piecewise_constant,
  "lr_policy_params": {
    "learning_rate": 0.1,
    "boundaries": [30, 60, 80, 90],
    "decay_rates": [0.1, 0.01, 0.001, 1e-4],
  },

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
    "output_dim": 30,
  },
  "loss": CrossEntropyLoss,
  "data_layer": SpeechCommandsDataLayer,
  "data_layer_params": {
    "dataset_location": "data/speech_commands_v0.01",
    "num_audio_features": 80,
    "num_labels": 30
  },
}

train_params = {
  "data_layer_params": {
    "dataset_files": [
      "training_list.txt"
    ],
    "shuffle": True,
  },
}

eval_params = {
  "data_layer_params": {
    "dataset_files": [
      "validation_list_labeled.txt"
    ],
    "shuffle": False,
  },
}