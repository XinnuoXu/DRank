#coding=utf8

if __name__ == '__main__':
	import sys
	import os
	import random
	pos_example = [line.strip() for line in open("coherence/dataset_readers/train_data.all/generated_dial_examples_train.pos")]
	neg_example = [line.strip() for line in open("coherence/dataset_readers/train_data.all/generated_dial_examples_train.neg")]
	for i in range(0, 10):
		random.shuffle(pos_example)
		random.shuffle(neg_example)
		for t_num in range(200, 1001, 200):
			fp_pos = open("coherence/dataset_readers/generated_dial_examples_train.pos", "w")
			fp_neg = open("coherence/dataset_readers/generated_dial_examples_train.neg", "w")
			for s in pos_example[:t_num]:
				fp_pos.write(s + "\n")
			for s in neg_example[:t_num]:
				fp_neg.write(s + "\n")
			fp_pos.close()
			fp_neg.close()
			test_pos = "coherence/dataset_readers/AMT_test.pos"
			test_neg = "coherence/dataset_readers/AMT_test.neg"
			os.system("srun --gres=gpu --partition=amd-longq --nodes 1 sh discriminator_pretrain.sh")
			os.system("srun --gres=gpu --partition=amd-longq --nodes 1 sh discriminator_test.sh " + test_pos + " " + test_neg)
			os.system("mv test.res test.res." + str(i) + "." + str(t_num))
			
