#coding=utf8

if __name__ == '__main__':
	import os
	import sys
	import random
	from discriminator import Discriminator
	from generator import Generator

	train_dev_rate = 0.8
	path_discriminator = "./discriminator/"
	file_train_pos = "coherence/dataset_readers/generated_dial_examples_train.pos"
	file_train_neg = "coherence/dataset_readers/generated_dial_examples_train.neg"
	file_dev_pos = "coherence/dataset_readers/generated_dial_examples_dev.pos"
	file_dev_neg = "coherence/dataset_readers/generated_dial_examples_dev.neg"
	path_train_pos = path_discriminator + file_train_pos
	path_train_neg = path_discriminator + file_train_neg
	path_dev_pos = path_discriminator + file_dev_pos
	path_dev_neg = path_discriminator + file_dev_neg

	if sys.argv[1] == "data":
		'''
		file_list = []
		for filename in os.listdir("./generator/"):
			if filename.startswith("dialogue-model.epoch") and filename.endswith(".ex"):
				file_list.append(filename.replace(".ex", ""))
		for filename in os.listdir("./generator/"):
			if filename.startswith("dialogue-model.epoch") and (not filename.endswith(".ex")):
				if filename not in file_list:
					os.system("nohup python generator.py " + filename + " &")
		'''
		os.system("nohup python generator.py dialogue-model.with_gold &")

	if sys.argv[1] == "pretrain":
		gen = Generator()
		neg_examples = [line.strip() for line in open("generator/dialogue-model.with_gold.ex")]
		pos_examples = ["\t".join(l) for l in gen.gold_dials]
		train_size = int(len(pos_examples) * train_dev_rate)

		random.shuffle(pos_examples)
		random.shuffle(neg_examples)

		open(path_train_pos, "w").write("\n".join(pos_examples[:train_size]) + "\n")
		open(path_dev_pos, "w").write("\n".join(pos_examples[train_size:]) + "\n")
		open(path_train_neg, "w").write("\n".join(neg_examples[:train_size]) + "\n")
		open(path_dev_neg, "w").write("\n".join(neg_examples[train_size:]) + "\n")

		os.system("cd " + path_discriminator + "; sh discriminator_pretrain.sh")

	if sys.argv[1] == "train":
		neg_examples = []
		gen = Generator()
		num_of_neg = 0
		for filename in os.listdir("./generator/"):
			if filename.startswith("dialogue-model.epoch") and filename.endswith(".ex"):
				gen_len = int(filename.split(".")[-2].replace("gen", ""))
				if gen_len > 2 and gen_len < 8:
					neg_examples += [line.strip() for line in open("./generator/" + filename)]
					num_of_neg += 1
		train_size = int(len(neg_examples) * train_dev_rate)
		random.shuffle(neg_examples)
		open(path_train_neg, "w").write("\n".join(neg_examples[:train_size]) + "\n")
		open(path_dev_neg, "w").write("\n".join(neg_examples[train_size:]) + "\n")

		pos_examples = ["\t".join(l) for l in gen.gold_dials]
		train_size = int(len(pos_examples) * train_dev_rate)
		random.shuffle(pos_examples)

		pos_train = []
		pos_dev = []
		for i in range(num_of_neg):
			pos_train += pos_examples[:train_size]
			pos_dev += pos_examples[train_size:]
		open(path_train_pos, "w").write("\n".join(pos_train) + "\n")
		open(path_dev_pos, "w").write("\n".join(pos_dev) + "\n")

		os.system("cd " + path_discriminator + "; sh discriminator_pretrain.sh")

	if sys.argv[1] == "test":
		disc = Discriminator()
		print ("ACC", disc.test())
