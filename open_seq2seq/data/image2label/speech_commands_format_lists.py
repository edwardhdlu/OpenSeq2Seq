import os

classes = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "bed", "bird", "cat", "dog", "eight", "five", "four", "happy", "house", "marvin", "nine", "one", "seven", "sheila", "six", "three", "tree", "two", "wow", "zero"]

root_dir = "../../../data/speech_commands_v0.01"
files = ["testing_list.txt", "validation_list.txt"]

for file in files:
	result = []
	lines = open(os.path.join(root_dir, file), "r").read().splitlines()

	for line in lines:
		label = line.split("/")[0]
		index = classes.index(label)
		result.append("{},{}".format(index, line))

	write_file = open(os.path.join(root_dir, file.split(".")[0] + "_labeled.txt"), "w")
	for line in result:
		write_file.write(line + "\n")