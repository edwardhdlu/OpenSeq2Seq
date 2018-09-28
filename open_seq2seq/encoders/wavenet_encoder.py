# Copyright (c) 2018 NVIDIA Corporation

import tensorflow as tf
from open_seq2seq.parts.cnns.conv_blocks import conv_actv
from open_seq2seq.parts.convs2s.utils import gated_linear_units

from .encoder import Encoder


def wavenet_conv_block(layer_type, name, inputs, condition, residual_channels, gate_channels, skip_channels, kernel_size, activation_fn, 
	strides, padding, regularizer, training, data_format, dilation, causal_convs=False, batch_norm=False, local_conditioning=False):

	# split input and conditioning in two
	input_shape = inputs.get_shape().as_list()
	input_filter = inputs[:, :, 0:int(input_shape[2] / 2)]
	input_gate = inputs[:, :, int(input_shape[2] / 2):]

	conv_filter = conv_actv(
		layer_type=layer_type,
		name="conv_filter_" + name,
		inputs=input_filter,
		filters=gate_channels,
		kernel_size=kernel_size,
		activation_fn=None,
		strides=strides,
		padding=padding,
		regularizer=regularizer,
		training=training,
		data_format=data_format,
		dilation=dilation
	)

	conv_gate = conv_actv(
		layer_type=layer_type,
		name="conv_gate_" + name,
		inputs=input_gate,
		filters=gate_channels,
		kernel_size=kernel_size,
		activation_fn=None,
		strides=strides,
		padding=padding,
		regularizer=regularizer,
		training=training,
		data_format=data_format,
		dilation=dilation
	)

	if causal_convs:
		conv_filter = tf.slice(conv_filter, [0, 0, 0], [-1, tf.shape(input_filter)[1] - (gate_channels - 1), -1])
		conv_gate = tf.slice(conv_gate, [0, 0, 0], [-1, tf.shape(input_gate)[1] - (gate_channels - 1), -1])

	if local_conditioning:
		input_shape_condition = condition.get_shape().as_list()
		input_filter_condition = condition[:, :, 0:int(input_shape_condition[2] / 2)]
		input_gate_condition = condition[:, :, int(input_shape_condition[2] / 2):]

		input_filter_condition = tf.expand_dims(input_filter_condition, 1)
		input_gate_condition = tf.expand_dims(input_gate_condition, 1)

		conv_filter_condition = tf.layers.conv2d_transpose(
			name="conv_filter_condition_" + name,
			inputs=input_filter_condition,
			filters=gate_channels,
			kernel_size=1, # 1x1 convolution
			strides=(1, 256), # scale factor
			kernel_regularizer=regularizer,
			data_format=data_format
		)

		conv_gate_condition = tf.layers.conv2d_transpose(
			name="conv_gate_condition" + name,
			inputs=input_gate_condition,
			filters=gate_channels,
			kernel_size=1, # 1x1 convolution
			strides=(1, 256), # scale factor
			kernel_regularizer=regularizer,
			data_format=data_format
		)

		conv_filter_condition = tf.squeeze(conv_filter_condition, [1])
		conv_filter = tf.pad(conv_filter, tf.constant([[0, 0], [0, 99], [0, 0]]))

		conv_gate_condition = tf.squeeze(conv_gate_condition, [1])

		conv_filter = tf.add(conv_filter, conv_filter_condition)
		conv_gate = tf.add(conv_gate, conv_gate_condition)
	
	conv_filter = tf.tanh(conv_filter)
	conv_gate = tf.sigmoid(conv_gate)
	product = tf.multiply(conv_filter, conv_gate)

	residual = conv_1x1(
		layer_type=layer_type,
		name="conv_residual_" + name,
		inputs=product,
		filters=residual_channels,
		strides=strides,
		regularizer=regularizer,
		training=training,
		data_format=data_format
	)

	skip = conv_1x1(
		layer_type=layer_type,
		name="conv_skip_" + name,
		inputs=product,
		filters=skip_channels,
		strides=strides,
		regularizer=regularizer,
		training=training,
		data_format=data_format
	)

	return residual, skip

def conv_1x1(layer_type, name, inputs, filters, strides, regularizer, training, data_format):
	block = conv_actv(
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

	return block

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
				"layers": int,
				"residual_channels": int,
				"gate_channels": int,
				"skip_channels": int,
				"output_channels": int,
			}
		)

	@staticmethod
	def get_optional_params():
		return dict(
			Encoder.get_optional_params(),
			**{
				"causal_convs": bool, # [TODO]
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
		* **layers** (int) --- total number of dilated causal convolutional layers
		
		* **residual_channels** (int) --- number of channels for block input and output
		* **gate_channels** (int) --- number of channels for the gated unit
		* **skip_channels** (int) --- number of channels for the skip output of the gated unit and skip connections 
		* **output_channels** (int) --- number of output channels
		"""

		super(WavenetEncoder, self).__init__(params, model, name, mode)

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

		source, src_length = input_dict["source_tensors"][0]
		condition, cond_length = input_dict["source_tensors"][1]

		# add dummy dimension
		source = tf.expand_dims(source, 2)

		training = (self._mode == "train")
		regularizer = self.params.get("regularizer", None)
		data_format = self.params.get("data_format", "channels_last")		

		if data_format == "channels_last":
			conv_feats = source
			spectrograms = condition
		else:
			conv_feats = tf.transpose(source, [0, 2, 1])
			spectrograms = tf.transpose(condition, [0, 2, 1])

		layer_type = self.params["layer_type"]
		kernel_size = self.params["kernel_size"]
		strides = self.params["strides"]
		padding = self.params["padding"]

		residual_channels = self.params["residual_channels"]
		gate_channels = self.params["gate_channels"]
		skip_channels = self.params["skip_channels"]
		output_channels = self.params["output_channels"]

		encoded_input = self._mu_law_encode(conv_feats, output_channels)
		print(encoded_input)

		conv_feats = tf.cast(encoded_input, self.params["dtype"])

		# ----- Convolutional layers -----------------------------------------------

		# [TODO] use conv_bn

		# causal layer
		conv_feats = conv_actv(
			layer_type=layer_type,
			name="causal_conv",
			inputs=conv_feats,
			filters=residual_channels, 
			kernel_size=kernel_size,
			activation_fn=None,
			strides=strides,
			padding=padding,
			regularizer=regularizer,
			training=training,
			data_format=data_format,
			dilation=1
		)

		blocks = self.params["blocks"]
		layers = self.params["layers"]
		layers_per_stack = int(layers / blocks)
		output_layer = None

		# dilation stack
		for layer in range(layers):
			dilation = 2 ** (layer % layers_per_stack)
			residual, skip = wavenet_conv_block(
				layer_type=layer_type, 
				name=str(layer + 1), 
				inputs=conv_feats, 
				condition=spectrograms,
				residual_channels=residual_channels, 
				gate_channels=gate_channels,
				skip_channels=skip_channels,
				kernel_size=kernel_size, 
				activation_fn=None, # unused 
				strides=strides, 
				padding=padding, 
				regularizer=regularizer, 
				training=training, 
				data_format=data_format, 
				dilation=dilation
			)

			conv_feats = tf.add(residual, conv_feats)

			if output_layer is not None: 
				output_layer = tf.add(skip, output_layer)
			else:
				output_layer = skip

		# skip-connections
		output_layer = tf.nn.relu(output_layer)
		output_layer = conv_1x1(
			layer_type=layer_type, 
			name="skip_1x1_1", 
			inputs=output_layer, 
			filters=skip_channels, 
			strides=1, 
			regularizer=regularizer, 
			training=training, 
			data_format=data_format
		)

		output_layer = tf.nn.relu(output_layer)
		output_layer = conv_1x1(
			layer_type=layer_type, 
			name="skip_1x1_2", 
			inputs=output_layer, 
			filters=output_channels, 
			strides=1, 
			regularizer=regularizer, 
			training=training, 
			data_format=data_format
		)

		return { "logits": output_layer, "outputs": [encoded_input] }
