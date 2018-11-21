# Produces a .csv containing file names for balanced classes and their labels
# Also randomly samples 1s clips from _background_noise_/ for silence label
import os
import random
import librosa

classes = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "bed", "bird", "cat", "dog", "eight", "five", "four", "happy", "house", "marvin", "nine", "one", "seven", "sheila", "six", "three", "tree", "two", "wow", "zero"]

samples_per_class = 1300
root_dir = "../../../data/speech_commands_v0.01"
result = []

# ensure our samples aren't in the test/validation sets
test_files = open(os.path.join(root_dir, "testing_list.txt")).read().splitlines()
val_files = open(os.path.join(root_dir, "validation_list.txt")).read().splitlines()
exclude = set(test_files + val_files)

# sample regular classes
for i, label in enumerate(classes):
	folder = os.path.join(root_dir, label)
	files = os.listdir(folder)
	class_samples = []

	for file in files:
		file_path = "{}/{}".format(label, file)
		if file_path not in exclude:
			class_samples.append("{},{}".format(i, file_path))

	result += class_samples[:samples_per_class]

random.shuffle(result)
print("Saving {} lines...".format(len(result)))

# save .csv file
write_file = open(os.path.join(root_dir, "training_list.txt"), "w")
for line in result:
	write_file.write(line + "\n")
