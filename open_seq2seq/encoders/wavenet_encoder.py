# Copyright (c) 2018 NVIDIA Corporation

import tensorflow as tf
from math import ceil
from open_seq2seq.parts.cnns.conv_blocks import conv_actv, conv_bn_actv
from open_seq2seq.parts.convs2s.utils import gated_linear_units

from .encoder import Encoder

def _get_receptive_field(kernel_size, blocks, layers_per_block):
  dilations = [2 ** i for i in range(layers_per_block)]
  return (kernel_size - 1) * blocks * sum(dilations) + 1

def _mu_law_encode(signal, channels):
  mu = tf.to_float(channels - 1)
  safe_audio_abs = tf.minimum(tf.abs(signal), 1.0)
  magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
  signal = tf.sign(signal) * magnitude
  return tf.to_int32((signal + 1) / 2 * mu + 0.5)
    
def _mu_law_decode(output, channels):
  mu = channels - 1
  signal = 2 * (tf.to_float(output) / mu) - 1
  magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
  return tf.sign(signal) * magnitude

def conv_1x1(layer_type, name, inputs, filters, strides, regularizer, training, data_format):

  return conv_actv(
    layer_type=layer_type,
    name=name,
    inputs=inputs,
    filters=filters,
    kernel_size=1,
    activation_fn=None,
    strides=strides,
    padding="SAME",
    regularizer=regularizer,
    training=training,
    data_format=data_format,
  )

def causal_conv_bn_actv(layer_type, name, inputs, filters, kernel_size, activation_fn, strides,
            padding, regularizer, training, data_format, bn_momentum, bn_epsilon, dilation=1):
  
  block = conv_bn_actv(
    layer_type=layer_type,
    name=name,
    inputs=inputs,
    filters=filters,
    kernel_size=kernel_size,
    activation_fn=activation_fn,
    strides=strides,
    padding=padding,
    regularizer=regularizer,
    training=training,
    data_format=data_format,
    bn_momentum=bn_momentum,
    bn_epsilon=bn_epsilon,
    dilation=dilation
  )

  # pad the left side of the time-series with an amount of zeros based on the dilation rate
  block = tf.pad(block, [[0, 0], [dilation * (kernel_size - 1), 0], [0,0]])
  return block

def wavenet_conv_block(layer_type, name, inputs, condition_filter, condition_gate, filters, kernel_size, activation_fn, strides,
        padding, regularizer, training, data_format, bn_momentum, bn_epsilon, layers_per_block, upsample_factor, 
        receptive_field):

  skips = None
  for layer in range(layers_per_block):
    # split source (raw audio) along channels
    source_shape = inputs.get_shape().as_list()
    source_filter = inputs[:, :, 0:int(source_shape[2] / 2)]
    source_gate = inputs[:, :, int(source_shape[2] / 2):]

    dilation = 2 ** layer

    source_filter = causal_conv_bn_actv(
      layer_type=layer_type,
      name="filter_{}_{}".format(name, layer),
      inputs=source_filter,
      filters=filters,
      kernel_size=kernel_size,
      activation_fn=None,
      strides=strides,
      padding=padding,
      regularizer=regularizer,
      training=training,
      data_format=data_format,
      bn_momentum=bn_momentum,
      bn_epsilon=bn_epsilon,
      dilation=dilation
    )

    source_gate = causal_conv_bn_actv(
      layer_type=layer_type,
      name="gate_{}_{}".format(name, layer),
      inputs=source_gate,
      filters=filters,
      kernel_size=kernel_size,
      activation_fn=None,
      strides=strides,
      padding=padding,
      regularizer=regularizer,
      training=training,
      data_format=data_format,
      bn_momentum=bn_momentum,
      bn_epsilon=bn_epsilon,
      dilation=dilation
    )

    if condition_filter is not None and condition_gate is not None:
      source_filter = tf.tanh(tf.add(source_filter, condition_filter))
      source_gate = tf.sigmoid(tf.add(source_gate, condition_gate))
    else:
      source_filter = tf.tanh(source_filter)
      source_gate = tf.sigmoid(source_gate)

    conv_feats = tf.multiply(source_filter, source_gate)

    residual = conv_1x1(
      layer_type=layer_type,
      name="residual_1x1_{}_{}".format(name, layer), 
      inputs=conv_feats, 
      filters=filters, 
      strides=strides, 
      regularizer=regularizer, 
      training=training, 
      data_format=data_format
    )

    skip = conv_1x1(
      layer_type=layer_type,
      name="skip_1x1_{}_{}".format(name, layer), 
      inputs=conv_feats, 
      filters=filters, 
      strides=strides, 
      regularizer=regularizer, 
      training=training, 
      data_format=data_format
    )

    inputs = tf.add(inputs, residual)

    if skips is None:
      skips = skip
    else:
      skips = tf.add(skips, skip)

  return inputs, skips

class WavenetEncoder(Encoder):

  """
  WaveNet like encoder.
  Fully convolutional.
  """

  @staticmethod
  def get_required_params():
    return dict(
      Encoder.get_required_params(),
      **{
        "layer_type": str,
        "kernel_size": int,
        "strides": int,
        "padding": str,
        "blocks": int,
        "layers_per_block": int,
        "activation_fn": None,
        "filters": int,
        "upsample_factor": int,
        "quantization_channels": int
      }
    )

  @staticmethod
  def get_optional_params():
    return dict(
      Encoder.get_optional_params(),
      **{
        "data_format": str,
        "bn_momentum": float,
        "bn_epsilon": float
      }
    )

  def __init__(self, params, model, name="wavenet_encoder", mode="train"):
    """
    WaveNet like encoder constructor.

    Config parameters:
    [TODO]
    * **layer_type** (str) --- type of layer, should be "conv1d"
    * **kernel_size** (int) --- size of the kernel
    * **strides** (int) --- size of stride
    * **padding** (str) --- padding, can be "SAME" or "VALID"

    * **blocks** (int) --- number of dilation cycles
    * **layers_per_block** (int) --- number of dilated causal convolutional layers in each block
    * **filters** (int) --- number of output channels
    """

    super(WavenetEncoder, self).__init__(params, model, name, mode)

  def _encode(self, input_dict):
    """
    Creates TensorFlow graph for WaveNet like encoder.
    ...
    """ 

    training = (self._mode == "train" or self._mode == "eval")

    # takes raw audio and spectrograms
    if training:
      source, src_length, spectrogram, spec_length = input_dict["source_tensors"]
      spec_offset = 0
    else:
      source, src_length, spectrogram, spec_length, spec_offset = input_dict["source_tensors"]

    regularizer = self.params.get("regularizer", None)
    data_format = self.params.get("data_format", "channels_last")   

    if data_format != "channels_last":
      source = tf.transpose(source, [0, 2, 1])
      spectrogram = tf.transpose(spectrogram, [0, 2, 1])

    condition = spectrogram

    layer_type = self.params["layer_type"]
    kernel_size = self.params["kernel_size"]
    strides = self.params["strides"]
    padding = self.params["padding"]
    blocks = self.params["blocks"]
    layers_per_block = self.params["layers_per_block"]
    activation_fn = self.params["activation_fn"]
    filters = self.params["filters"]
    upsample_factor = self.params["upsample_factor"]
    quantization_channels = self.params["quantization_channels"]

    bn_momentum = self.params.get("bn_momentum", 0.1)
    bn_epsilon = self.params.get("bn_epsilon", 1e-5)
    local_conditioning = self.params.get("local_conditioning", True)
    conv_upsampling = self.params.get("conv_upsampling", True)

    receptive_field = _get_receptive_field(kernel_size, blocks, layers_per_block)

    # ----- Preprocessing -----------------------------------------------

    encoded_inputs = _mu_law_encode(source, quantization_channels)

    if training:
      # remove last sample to maintain causality
      inputs = tf.slice(encoded_inputs, [0, 0], [-1, tf.shape(encoded_inputs)[1] - 1]) 
    else:
      inputs = encoded_inputs
    
    inputs = tf.one_hot(inputs, depth=quantization_channels, axis=-1)
    inputs = tf.cast(inputs, self.params["dtype"])

    if local_conditioning:
      # split condition (mel spectrograms) along channels
      condition_shape = condition.get_shape().as_list()
      condition_filter = condition[:, :, 0:int(condition_shape[2] / 2)]
      condition_gate = condition[:, :, int(condition_shape[2] / 2):]

      if conv_upsampling:
        condition_filter = tf.expand_dims(condition_filter, 1)
        for i in range(upsample_factor):
          condition_filter = tf.layers.conv2d_transpose(
            name="filter_condition_{}".format(i),
            inputs=condition_filter,
            filters=filters,
            kernel_size=1, # 1x1 convolution
            strides=(1, 2), # scale factor
            kernel_regularizer=regularizer,
            data_format=data_format
          )
        condition_filter = tf.squeeze(condition_filter, [1])
        
        condition_gate = tf.expand_dims(condition_gate, 1)
        for i in range(upsample_factor):
          condition_gate = tf.layers.conv2d_transpose(
            name="gate_condition_{}".format(i),
            inputs=condition_gate,
            filters=filters,
            kernel_size=1, # 1x1 convolution
            strides=(1, 2), # scale factor
            kernel_regularizer=regularizer,
            data_format=data_format
          )
        condition_gate = tf.squeeze(condition_gate, [1])

      else:
        condition_filter = conv_1x1(
          layer_type=layer_type,
          name="filter_condition", 
          inputs=condition_filter, 
          filters=filters, 
          strides=strides, 
          regularizer=regularizer, 
          training=training, 
          data_format=data_format
        )

        condition_gate = conv_1x1(
          layer_type=layer_type,
          name="gate_condition", 
          inputs=condition_gate, 
          filters=filters, 
          strides=strides, 
          regularizer=regularizer, 
          training=training, 
          data_format=data_format
        )

      if training:
        condition_filter = condition_filter[:, :-1, :]
        condition_gate = condition_gate[:, :-1, :]
      else:
        condition_filter = condition_filter[:, spec_offset:spec_offset + receptive_field, :]
        condition_gate = condition_gate[:, spec_offset:spec_offset + receptive_field, :]

    else:
      condition_filter = None
      condition_gate = None

    # ----- Convolutional layers -----------------------------------------------

    # first causal convolutional layer
    inputs = causal_conv_bn_actv(
      layer_type=layer_type,
      name="causal_input",
      inputs=inputs,
      filters=filters, 
      kernel_size=kernel_size,
      activation_fn=None,
      strides=strides,
      padding=padding,
      regularizer=regularizer,
      training=training,
      data_format=data_format,
      bn_momentum=bn_momentum,
      bn_epsilon=bn_epsilon,
      dilation=1
    )

    # dilation stack
    skips = None
    for block in range(blocks):
      # inputs = conv_1x1(
      #   layer_type=layer_type,
      #   name="adapter_1x1_{}".format(block), 
      #   inputs=inputs, 
      #   filters=filters, 
      #   strides=strides, 
      #   regularizer=regularizer, 
      #   training=training, 
      #   data_format=data_format
      # )

      inputs, skip = wavenet_conv_block(
        layer_type=layer_type,
        name=block,
        inputs=inputs,
        condition_filter=condition_filter,
        condition_gate=condition_gate,
        filters=filters,
        kernel_size=kernel_size,
        activation_fn=None,
        strides=strides,
        padding=padding,
        regularizer=regularizer,
        training=training,
        data_format=data_format,
        bn_momentum=bn_momentum,
        bn_epsilon=bn_epsilon,
        layers_per_block=layers_per_block,
        upsample_factor=upsample_factor,
        receptive_field=receptive_field
      )

      if skips is None:
        skips = skip
      else:
        skips = tf.add(skips, skip)

    # postprocessing (outputs)
    outputs = tf.nn.relu(skips)
    outputs = conv_1x1(
      layer_type=layer_type,
      name="postprocess_1", 
      inputs=outputs, 
      filters=filters, 
      strides=strides, 
      regularizer=regularizer, 
      training=training, 
      data_format=data_format
    )

    outputs = tf.nn.relu(outputs)
    outputs = conv_1x1(
      layer_type=layer_type,
      name="postprocess_2", 
      inputs=outputs, 
      filters=quantization_channels, 
      strides=strides, 
      regularizer=regularizer, 
      training=training, 
      data_format=data_format
    )

    if training:
      prediction = tf.slice(outputs, [0, receptive_field - 1, 0], [-1, -1, -1])
      target_output = tf.slice(encoded_inputs, [0, receptive_field], [-1, -1])
    else:
      prediction = outputs
      target_output = encoded_inputs

    audio = tf.argmax(tf.nn.softmax(outputs), axis=-1, output_type=tf.int32)
    audio = tf.expand_dims(audio, -1)
    audio = _mu_law_decode(audio, self.params["quantization_channels"])
    audio = tf.cast(audio, tf.float32)

    return { "logits": prediction, "outputs": [target_output, audio] }
