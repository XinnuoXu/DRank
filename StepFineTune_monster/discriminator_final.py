#coding=utf8

if __name__ == '__main__':
	import os
	import sys
	import random
	from discriminator import Discriminator
	from generator import Generator

	num_of_samples = 2000
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
		file_list = []
		for filename in os.listdir("./generator/"):
			if filename.startswith("dialogue-model.epoch") and filename.endswith(".ex"):
				file_list.append(filename.replace(".ex", ""))
		for filename in os.listdir("./generator/"):
			if filename.startswith("dialogue-model.epoch") and (not filename.endswith(".ex")):
				if filename not in file_list:
					#os.system("nohup python generator.py " + filename + " &")
					for i in range(0, 3):
						os.system("nohup srun --partition amd-longq --nodes 1 --gres=gpu python generator.py " + filename + " " + str(i) + " &")
		#os.system("nohup python generator.py dialogue-model.with_gold &")

	if sys.argv[1] == "train":
		for turn_num in range(2, 3):
			gen = Generator()
			neg_examples_dict = {}
			neg_examples = []
			for filename in os.listdir("./generator/"):
				if filename.startswith("dialogue-model.epoch") and filename.endswith(".ex"):
					gen_len = int(filename.split(".")[-3].replace("gen", ""))
					log_id = int(filename.split(".")[-2])
					if gen_len >= 0 and gen_len < 10:
						if gen_len not in neg_examples_dict:
							neg_examples_dict[gen_len] = []
						neg_examples_dict[gen_len] += [line.strip() for line in open("./generator/" + filename)]
			for item in neg_examples_dict:
				neg_examples += neg_examples_dict[item]
			#neg_examples = random.sample(neg_examples, num_of_samples)
			train_neg_size = int(len(neg_examples) * train_dev_rate)
			random.shuffle(neg_examples)
			open(path_train_neg, "w").write("\n".join(neg_examples[:train_neg_size]) + "\n")
			open(path_dev_neg, "w").write("\n".join(neg_examples[train_neg_size:]) + "\n")

			pos_examples = ["\t".join(l) for l in gen.gold_dials]
			train_size = int(len(pos_examples) * train_dev_rate)
			random.shuffle(pos_examples)

			pos_train = []
			pos_dev = []
			while len(pos_train) < train_neg_size:
				pos_train += pos_examples[:train_size]
				pos_dev += pos_examples[train_size:]
			open(path_train_pos, "w").write("\n".join(pos_train) + "\n")
			open(path_dev_pos, "w").write("\n".join(pos_dev) + "\n")

			#os.system("cd " + path_discriminator + "; sh discriminator_pretrain.sh")
			os.system("cd " + path_discriminator + "; srun --partition amd-longq --nodes 1 --gres=gpu sh discriminator_pretrain.sh " + str(turn_num))

	if sys.argv[1] == "test":
		for i in range(2, 3):
			disc = Discriminator()
			print ("ACC", disc.test(i))
