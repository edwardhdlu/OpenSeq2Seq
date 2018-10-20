# pylint: skip-file
import tensorflow as tf
from open_seq2seq.models import Text2SpeechWavenet
from open_seq2seq.encoders import WavenetEncoder
from open_seq2seq.encoders.wavenet_encoder import _get_receptive_field
from open_seq2seq.decoders import FakeDecoder
from open_seq2seq.losses import WavenetLoss
from open_seq2seq.data import WavenetDataLayer
from open_seq2seq.optimizers.lr_policies import exp_decay
from open_seq2seq.parts.convs2s.utils import gated_linear_units

base_model = Text2SpeechWavenet

kernel_size = 2
blocks = 4
layers_per_block = 6
receptive_field = _get_receptive_field(kernel_size, blocks, layers_per_block)

base_params = {
  "random_seed": 0,
  "use_horovod": True,
  "max_steps": 1000000,

  "num_gpus": 8,
  "batch_size_per_gpu": 1,

  "save_summaries_steps": 50,
  "print_loss_steps": 50,
  "print_samples_steps": 500,
  "eval_steps": 500,
  "save_checkpoint_steps": 2500,
  "logdir": "/results/WAVENET-TRAIN",

  "optimizer": "Adam",
  "optimizer_params": {},
  "lr_policy": exp_decay,
  "lr_policy_params": {
    "learning_rate": 1e-3,
    "decay_steps": 20000,
    "decay_rate": 0.1,
    "use_staircase_decay": False,
    "begin_decay_at": 45000,
    "min_lr": 1e-5,
  },
  "dtype": tf.float32,
  "regularizer": tf.contrib.layers.l2_regularizer,
  "regularizer_params": {
    "scale": 1e-6
  },
  "initializer": tf.contrib.layers.xavier_initializer,

  "summaries": [],

  "encoder": WavenetEncoder,
  "encoder_params": {
    "layer_type": "conv1d",
    "kernel_size": kernel_size,
    "strides": 1,
    "padding": "VALID",
    "blocks": blocks,
    "layers_per_block": layers_per_block,
    "activation_fn": gated_linear_units,
    "filters": 64,
    "upsample_factor": 8, 
    "quantization_channels": 256
  },

  "decoder": FakeDecoder,

  "loss": WavenetLoss,
  "loss_params": {
    "receptive_field": receptive_field
  },

  "data_layer": WavenetDataLayer,
  "data_layer_params": {
    "dataset": "LJ",
    "num_audio_features": 80,
    "dataset_location": "/data/LJSpeech-1.1-partitioned/wavs/"
  }
}

train_params = {
  "data_layer_params": {
    "dataset_files": [
      "/data/LJSpeech-1.1-partitioned/train.csv",
    ],
    "shuffle": True,
  },
}

eval_params = {
  "data_layer_params": {
    "dataset_files": [
      "/data/LJSpeech-1.1-partitioned/val.csv",
    ],
    "shuffle": False,
  },
}

infer_params = {
  "data_layer_params": {
    "dataset_files": [
      "/data/LJSpeech-1.1-partitioned/test.csv",
    ],
    "shuffle": False,
  },
}

interactive_infer_params = {
  "data_layer_params": {
    "dataset_files": [],
    "shuffle": False,
  },
}