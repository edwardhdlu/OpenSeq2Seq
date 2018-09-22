# Copyright (c) 2018 NVIDIA Corporation
import tensorflow as tf

from open_seq2seq.data.data_layer import DataLayer
from open_seq2seq.data.text2speech.speech_utils import get_speech_features_from_file as get_spectrograms_from_file
from open_seq2seq.data.speech2text.speech_utils import get_speech_features_from_file as get_audio_from_file

class WavenetDataLayer(DataLayer):
	""" Text to speech data layer class for Wavenet """

	@staticmethod
	def get_required_params():
		return dict(
			DataLayer.get_required_params(), **{
				"dataset": ["LJ"],
			}
		)

	@staticmethod
	def get_optional_params():
		return dict(
			DataLayer.get_optional_params(), **{
				"dataset_location": str
			}
		)

	def __init__(self, params, model, num_workers=None, worker_id=None):
		"""
		Wavenet data layer constructor.

		See parent class for arguments description.

		Config parameters:

		* **dataset** (str) --- The dataset to use, currently only supports "LJ" for LJSpeech 1.1

		"""

		super(WavenetDataLayer, self).__init__(
			params,
			model,
			num_workers,
			worker_id
		)

		if self.params.get("dataset_location", None) is None:
			raise ValueError(
				"dataset_location must be specified when using LJSpeech"
			)

		names = ["wav_filename", "raw_transcript", "transcript"]
		sep = "\x7c"
		header = None

		self._sampling_rate = 22050
		self._n_fft = 1024

		n_mels = self.params["num_audio_features"]

		self._files = None
		for csvs in params["datset_files"]:
			files = pd.read_csv(
				csvs,
				encoding="utf-8",
				sep=sep,
				header=header,
				names=names,
				quoting=3
			)

			if self._files is None:
				self._files = files
			else:
				self._files self._files.append(files)

		cols = "wav_filename"
		all_files = self._files.loc[:, cols].values
		self._files = self.split_data(all_files)

		self._size = self.get_size_in_samples()
		self._dataset = None
		self._iterator = None
		self._input_tensors = None

	@property
	def input_tensors(self):
		return self._input_tensors

	def next_batch_feed_dict(self):
		print "TODO"

	def get_size_in_samples(self):
		return len(self._files)



	@property
	def iterator(self):
		return self._iterator

	def _parse_audio_element(self, element):
		"""Parses tf.data element from TextLineDataset into audio."""
		audio_filename, transcript = element

		if six.PY2:
			audio_filename = unicode(audio_filename, "utf-8")
		else:
			audio_filename = str(audio_filename, "utf-8")

		file_path = os.path.join(self.params["dataset_location"], audio_filename + ".wav")

		audio, spectrogram = get_speech_features_from_file(
			file_path,
			self.params["num_audio_features"],
			features_type="mels",
			return_raw_audio=True
		)

		# [TODO] add padding here?

		return 
			audio.astype(self.params["dtype"].as_numpy_dtype()), \
			np.int32([len(audio)]), \
			spectrogram.astype(self.params["dtype"].as_numpy_dtype()), \
			np.int32([len(spectrogram)])

	def build_graph(self):
		""" builds data reading graph """
		self._dataset = tf.data.Dataset.from_tensor_slices(self._files)

		if self.params["shuffle"]:
			self._dataset = self._dataset.shuffle(self._size)
		self._dataset = self._dataset.repeat()

		num_audio_features = self.params["num_audio_features"]

		if self.params["mode"] != "infer":
			self._dataset = self._dataset.map(
				lambda line: tf.py_func(
					self._parse_audio_element,
					[line],
					[self.params["dtype"], tf.int32, self.params["dtype"], tf.int32], 
					stateful=False
				),
				num_parallel_calls=8
			)

			# [TODO] add duration filters?
			# [TODO] add padding?

		else:
			print("[TODO] support inference")

		self._iterator = self._dataset.prefetch(tf.contrib.data.AUTOTUNE.make_initializable_iterator())

		if self.params["mode"] != "infer":
			source, src_length, spec, spec_length = self._iterator.get_next()
			spec.set_shape([self.params["batch_size"], None, num_audio_features])
			spec_length = tf.reshape(spec_length, [self.params["batch_size"]])

		else:
			print("[TODO] support inference")

		self._input_tensors = {}
		self._input_tensors["source_tensors"] = [source, src_length]
		self._input_tensors["condition_tensors"] = [spec, spec_length]