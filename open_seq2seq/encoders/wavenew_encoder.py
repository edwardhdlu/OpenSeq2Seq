# Copyright (c) 2018 NVIDIA Corporation

import tensorflow as tf
from math import ceil
from open_seq2seq.parts.cnns.conv_blocks import conv_actv, conv_bn_actv
from open_seq2seq.parts.convs2s.utils import gated_linear_units

from .encoder import Encoder

class WavenewEncoder(Encoder):

  """
  WaveNet like encoder.
  Fully convolutional.
  """

  @staticmethod
  def get_required_params():
    return dict(
      Encoder.get_required_params(),
      **{
        "activation_fn": None,
        'dropout_keep_prob': float,
        'convnet_layers': list,
        "quantization_channels": int
      }
    )

  @staticmethod
  def get_optional_params():
    return dict(
      Encoder.get_optional_params(),
      **{
        "padding": str,
        'data_format': ['channels_first', 'channels_last'],
        "bn_momentum": float,
        "bn_epsilon": float
      }
    )

  def __init__(self, params, model, name="wavenet_encoder", mode="train"):
    """
    WaveNet like encoder constructor.

    Config parameters:
    * **layer_type** (str) --- type of layer, should be "conv1d"
    * **kernel_size** (int) --- size of the kernel
    * **strides** (int) --- size of stride
    * **padding** (str) --- padding, can be "SAME" or "VALID"

    * **blocks** (int) --- number of dilation cycles
    * **layers_per_block** (int) --- number of dilated causal convolutional layers in each block
    * **filters** (int) --- number of output channels
    """

    super(WavenewEncoder, self).__init__(params, model, name, mode)

  def _mu_law_encode(self, signal, channels):
    mu = tf.to_float(channels - 1)
    safe_audio_abs = tf.minimum(tf.abs(signal), 1.0)
    magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
    signal = tf.sign(signal) * magnitude

    return tf.to_int32((signal + 1) / 2 * mu + 0.5)

  def _encode(self, input_dict):
    """
    Creates TensorFlow graph for WaveNet like encoder.
    ...
    """

    # takes raw audio and spectrograms
    #source, src_length = input_dict["source_tensors"][0]
    #spectrogram, spec_length = input_dict["source_tensors"][1]
    source, src_length, spectrogram, spec_length = input_dict["source_tensors"]

    # add dummy dimension to raw audio (1 channel)
    source = tf.expand_dims(source, 2)

    training = (self._mode == "train")
    regularizer = self.params.get("regularizer", None)
    data_format = self.params.get("data_format", "channels_last")
    convnet_layers = self.params["convnet_layers"]
    dropout_keep_prob = self.params["dropout_keep_prob"]

    if data_format != "channels_last":
      # source = tf.transpose(source, [0, 2, 1])
      spectrogram = tf.transpose(spectrogram, [0, 2, 1])

    padding = self.params.get("padding", "SAME")
    bn_momentum = self.params.get("bn_momentum", 0.1)
    bn_epsilon = self.params.get("bn_epsilon", 1e-5)

    # print(source.get_shape())
    encoded_input = self._mu_law_encode(source, self.params["quantization_channels"])
    # print(encoded_input.get_shape())
    # inputs = tf.cast(encoded_input, self.params["dtype"])
    inputs = spectrogram

    # ----- Convolutional layers -----------------------------------------------

    # preprocessing causal convolutional layer
    inputs = conv_bn_actv(
        layer_type="conv1d",
        name="preprocess_1",
        inputs=inputs,
        filters=128,
        kernel_size=1,
        activation_fn=self.params['activation_fn'],
        strides=1,
        padding=padding,
        regularizer=regularizer,
        training=training,
        data_format=data_format,
        bn_momentum=self.params.get('bn_momentum', 0.1),
        bn_epsilon=self.params.get('bn_epsilon', 1e-5),
    )

    inputs = tf.expand_dims(inputs, 1)

    for idx_convnet in range(len(convnet_layers)):
      ch_out = convnet_layers[idx_convnet]['num_channels']
      kernel_size = convnet_layers[idx_convnet]['kernel_size']
      strides = convnet_layers[idx_convnet]['stride']
      dropout_keep = convnet_layers[idx_convnet].get(
          'dropout_keep_prob', dropout_keep_prob) if training else 1.0

      inputs = tf.layers.conv2d_transpose(
          inputs,
          ch_out,
          kernel_size,
          strides=strides,
          padding=padding,
          data_format=data_format,
          kernel_regularizer=regularizer,
          name="conv_{}".format(idx_convnet + 1)
      )

      inputs = tf.layers.batch_normalization(
          name="bn_{}".format(idx_convnet + 1),
          inputs=inputs,
          gamma_regularizer=regularizer,
          training=training,
          axis=-1 if data_format == 'channels_last' else 1,
          momentum=bn_momentum,
          epsilon=bn_epsilon,
      )

      inputs = self.params['activation_fn'](inputs)

      inputs = tf.nn.dropout(x=inputs, keep_prob=dropout_keep)

    inputs = tf.squeeze(inputs, 1)
    # postprocessing (outputs)
    outputs = conv_bn_actv(
        layer_type="conv1d",
        name="postprocess_1",
        inputs=inputs,
        filters=512,
        kernel_size=1,
        activation_fn=self.params['activation_fn'],
        strides=1,
        padding=padding,
        regularizer=regularizer,
        training=training,
        data_format=data_format,
        bn_momentum=self.params.get('bn_momentum', 0.1),
        bn_epsilon=self.params.get('bn_epsilon', 1e-5),
    )

    outputs = conv_bn_actv(
        layer_type="conv1d",
        name="postprocess_2",
        inputs=outputs,
        filters=self.params["quantization_channels"],
        kernel_size=1,
        activation_fn=self.params['activation_fn'],
        strides=1,
        padding=padding,
        regularizer=regularizer,
        training=training,
        data_format=data_format,
        bn_momentum=self.params.get('bn_momentum', 0.1),
        bn_epsilon=self.params.get('bn_epsilon', 1e-5),
    )
    audio = tf.nn.softmax(outputs)

    return {"logits": outputs, "outputs": [encoded_input, audio] }
