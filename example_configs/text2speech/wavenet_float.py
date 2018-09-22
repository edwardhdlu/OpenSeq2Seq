# pylint: skip-file
import tensorflow as tf
from open_seq2seq.models import Text2SpeechWavenet
from open_seq2seq.encoders import WavenetEncoder

base_model = Text2SpeechWavenet

base_params = {
	"random_seed": 0,
	"use_horovod": False,
	"max_steps": 10000,

	"num_gpus": 1,
	"batch_size_per_gpu": 32,

	# [TODO] add logging params

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
	}
	"initializer": tf.contrib.layers.xavier_initializer,

	"summaries": [],

	"encoder": WavenetEncoder,
	"encoder_params": {
		"layer_type": "conv1d",
		"kernel_size": 3,
		"stride": 1,
		"padding": "VALID",
		"blocks": 4,
		"layers": 24,
		"residual_channels": 512,
		"gate_channels": 512,
		"skip_channels": 256,
		"output_channels": 30,
		"quantization_channels": 256
	}
}

train_params = {
	"data_layer_params": {
		"dataset_files": [
			"/data/speech/LJSpeech/train.csv",
		],
		"shuffle": True,
	},
}

eval_params = {
	"data_layer_params": {
		"dataset_files": [
			"/data/speech/LJSpeech/val.csv",
		],
		"shuffle": False,
	},
}

infer_params = {
	"data_layer_params": {
		"dataset_files": [
			"/data/speech/LJSpeech/test.csv",
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
