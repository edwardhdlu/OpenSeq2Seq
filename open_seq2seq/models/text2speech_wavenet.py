# Copyright (c) 2018 NVIDIA Corporation
import tensorflow as tf

from .encoder_decoder import EncoderDecoderModel

class Text2SpeechWavenet(EncoderDecoderModel):
	# [TODO] add logging info

	@staticmethod
	def get_required_params():
		return dict(
			EncoderDecoderModel.get_required_params(), **{
				# "key": int,
			}
		)

	def __init__(self, params, mode="train", hvd=None):
		super(Text2SpeechWavenet, self).__init__(params, mode=mode, hvd=hvd)

	def evaluate(self, input_values, output_values):
		# reduce data for Horovod?
		return [input_values, output_values]

	def finalize_evaluation(self, results_per_batch, training_step=None):
		return {}
		