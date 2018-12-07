# pylint: skip-file
import tensorflow as tf
from open_seq2seq.models import Image2Label
from open_seq2seq.encoders import TDNNEncoder
from open_seq2seq.decoders import FullyConnectedSCDecoder
from open_seq2seq.data.speech2text.speech_commands import SpeechCommandsDataLayer
from open_seq2seq.losses import CrossEntropyLoss
from open_seq2seq.optimizers.lr_policies import poly_decay

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
    "use_horovod": True,
    "num_epochs": 10,

    "num_gpus": 1,
    "batch_size_per_gpu": 64,
    "iter_size": 1,

    "save_summaries_steps": 1000,
    "print_loss_steps": 10,
    "print_samples_steps": 5000,
    "eval_steps": 100,
    "save_checkpoint_steps": 5000,
    "logdir": "result_jasper/speech_commands_float",

    "optimizer": "Momentum",
    "optimizer_params": {
        "momentum": 0.90,
    },
    "lr_policy": poly_decay,
    "lr_policy_params": {
        "learning_rate": 0.01,
        "min_lr": 1e-5,
        "power": 2.0,
    },
    "larc_params": {
        "larc_eta": 0.001,
    },

    "regularizer": tf.contrib.layers.l2_regularizer,
    "regularizer_params": {
        'scale': 0.001
    },

    "dtype": tf.float32,
    # "loss_scaling": "Backoff",

    "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                  'variable_norm', 'gradient_norm', 'global_gradient_norm'],

    "encoder": TDNNEncoder,
    "encoder_params": {
        "convnet_layers": [
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [11], "stride": [2],
                "num_channels": 256, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8,
            },
            {
                "type": "conv1d", "repeat": 3,
                "kernel_size": [11], "stride": [1],
                "num_channels": 256, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8,
                "residual": True
            },
            {
                "type": "conv1d", "repeat": 3,
                "kernel_size": [11], "stride": [1],
                "num_channels": 256, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8,
                "residual": True
            },
            {
                "type": "conv1d", "repeat": 3,
                "kernel_size": [13], "stride": [1],
                "num_channels": 384, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8,
                "residual": True
            },
            {
                "type": "conv1d", "repeat": 3,
                "kernel_size": [13], "stride": [1],
                "num_channels": 384, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8,
                "residual": True
            },
            {
                "type": "conv1d", "repeat": 3,
                "kernel_size": [17], "stride": [1],
                "num_channels": 512, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8,
                "residual": True
            },
            {
                "type": "conv1d", "repeat": 3,
                "kernel_size": [17], "stride": [1],
                "num_channels": 512, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8,
                "residual": True
            },
            {
                "type": "conv1d", "repeat": 3,
                "kernel_size": [21], "stride": [1],
                "num_channels": 640, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.7,
                "residual": True
            },
            {
                "type": "conv1d", "repeat": 3,
                "kernel_size": [21], "stride": [1],
                "num_channels": 640, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.7,
                "residual": True
            },
            {
                "type": "conv1d", "repeat": 3,
                "kernel_size": [25], "stride": [1],
                "num_channels": 768, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.7,
                "residual": True
            },
            {
                "type": "conv1d", "repeat": 3,
                "kernel_size": [25], "stride": [1],
                "num_channels": 768, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.7,
                "residual": True
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [29], "stride": [1],
                "num_channels": 896, "padding": "SAME",
                "dilation":[2], "dropout_keep_prob": 0.6,
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [1], "stride": [1],
                "num_channels": 1024, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.6,
            }
        ],

        "dropout_keep_prob": 0.7,

        "initializer": tf.contrib.layers.xavier_initializer,
        "initializer_params": {
            'uniform': False,
        },
        "normalization": "batch_norm",
        "activation_fn": lambda x: tf.minimum(tf.nn.relu(x), 20.0),
        "data_format": "channels_last",
    },

    "decoder": FullyConnectedSCDecoder,
    "decoder_params": {
        "output_dim": num_labels,
    },

    "loss": CrossEntropyLoss,
    "data_layer": SpeechCommandsDataLayer,
    "data_layer_params": {
        "dataset_location": dataset_location,
        "num_audio_features": 80,
        "num_labels": num_labels,
        "cache_data": True,
        "augment_data": True
    },
}

train_params = {
  "data_layer_params": {
    "dataset_files": [
      dataset_version + "-train.txt"
    ],
    "shuffle": True,
    "repeat": True
  },
}

eval_params = {
  "batch_size_per_gpu": 16,
  "data_layer_params": {
    "dataset_files": [
      dataset_version + "-val.txt"
    ],
    "shuffle": False,
    "repeat": False
  },
}