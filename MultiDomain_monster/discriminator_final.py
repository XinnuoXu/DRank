#coding=utf8

if __name__ == '__main__':
	import os
	import sys
	import random
	from discriminator import Discriminator
	from generator import Generator

	num_of_samples = 2200
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
		for i in range(0, 30):
			filename = "dialogue-model_best.pt"
			os.system("nohup srun --partition amd-longq --nodes 1 --gres=gpu python generator.py " +filename+" "+str(i)+" &")

	if sys.argv[1] == "train":
		for turn_num in range(1, 10):
			neg_examples = []
			gen = Generator()
			for filename in os.listdir("./generator/"):
				if filename.startswith("dialogue-model") and filename.endswith(".ex"):
					neg_examples += [line.strip() for line in open("./generator/" + filename)]
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
			os.system("cd " + path_discriminator + "; srun --partition amd-shortq --nodes 1 --gres=gpu sh discriminator_pretrain.sh " + str(turn_num))

	if sys.argv[1] == "test":
		for i in range(9, 10):
			disc = Discriminator()
			print ("ACC", disc.test(i))
