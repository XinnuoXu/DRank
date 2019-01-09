import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import pdb
import os
import random

class Discriminator():

    def __init__(self):
        self.train_dev_rate = 0.8
        self.path_discriminator = "./discriminator/"
        self.file_train_pos = "coherence/dataset_readers/generated_dial_examples_train.pos"
        self.file_train_neg = "coherence/dataset_readers/generated_dial_examples_train.neg"
        self.file_dev_pos = "coherence/dataset_readers/generated_dial_examples_dev.pos"
        self.file_dev_neg = "coherence/dataset_readers/generated_dial_examples_dev.neg"
        self.path_train_pos = self.path_discriminator + self.file_train_pos
        self.path_train_neg = self.path_discriminator + self.file_train_neg
        self.path_dev_pos = self.path_discriminator + self.file_dev_pos
        self.path_dev_neg = self.path_discriminator + self.file_dev_neg
        self.path_test = "coherence/dataset_readers/generated_dial_examples_test"

        self.path_AMT_dev_pos = "testset/generated_dial_examples_dev.pos"
        self.path_AMT_dev_neg = "testset/generated_dial_examples_dev.neg"
        self.path_AMT_test_pos = "../../ranking_testset/AMT_test.pos"
        self.path_AMT_test_neg = "../../ranking_testset/AMT_test.neg"

        self.path_cls_res = self.path_discriminator + "/test.res"
        self.accumulate_neg_examples = self.path_discriminator + "/trained_models/neg.examples"

        self.trained_models = self.path_discriminator + "/trained_models"
        self.pretrained_models = self.path_discriminator + "/trained_models.pretrain"

    def re_fresh(self):
        os.system("rm -rf " + self.trained_models)
        os.system("cp -r " + self.pretrained_models + " " + self.trained_models)

    def train(self, pos_examples, neg_examples):
        train_size = int(len(pos_examples) * self.train_dev_rate)
        open(self.path_train_pos, "w").write("\n".join(pos_examples[:train_size]) + "\n")
        open(self.path_dev_pos, "w").write("\n".join(pos_examples[train_size:]) + "\n")

        '''
        # filter neg examples
        filterd_neg = []
        for example in neg_examples:
            flist = example.split("\t")
            if flist[-1].find("thank you") > -1:
                continue
            filterd_neg.append(example)
        train_size = int(len(filterd_neg) * self.train_dev_rate)
        '''

        '''
        open(self.accumulate_neg_examples, "a").write("\n".join(neg_examples) + "\n")
        all_negs = [line.strip() for line in open(self.accumulate_neg_examples, "r")]
        sample_size = len(pos_examples)
        filterd_neg = random.sample(all_negs, sample_size)
        train_size = int(len(filterd_neg) * self.train_dev_rate)
        '''

        open(self.path_train_neg, "w").write("\n".join(neg_examples[:train_size]) + "\n")
        open(self.path_dev_neg, "w").write("\n".join(neg_examples[train_size:]) + "\n")

        os.system("cd " + self.path_discriminator + "; sh discriminator_fine_tune.sh")
        return self.dev_epoch()

    def sen_by_sen(self, llist, given_turns):
        sentence_list = []
        for i in range(len(llist)):
            senlist = llist[i].split("\t")
            for j in range(given_turns + 2, len(senlist)+1, 2):
                sentence_list.append("\t".join(senlist[:j]))
        return sentence_list

    def pre_train(self, pos_examples, neg_examples, given_turns):
        train_size = int(len(pos_examples) * self.train_dev_rate)
        open(self.path_train_pos, "w").write("\n".join(self.sen_by_sen(pos_examples[:train_size], given_turns)) + "\n")
        open(self.path_dev_pos, "w").write("\n".join(self.sen_by_sen(pos_examples[train_size:], given_turns)) + "\n")
        open(self.path_train_neg, "w").write("\n".join(self.sen_by_sen(neg_examples[:train_size], given_turns)) + "\n")
        open(self.path_dev_neg, "w").write("\n".join(self.sen_by_sen(neg_examples[train_size:], given_turns)) + "\n")
        os.system("cd " + self.path_discriminator + "; sh discriminator_pretrain.sh")
        open(self.accumulate_neg_examples, "w").write("\n".join(self.sen_by_sen(neg_examples, given_turns)) + "\n")

    def batchClassify(self, examples):
        open(self.path_discriminator + self.path_test, "w").write("\n".join(examples) + "\n")
        os.system("cd " + self.path_discriminator + "; sh discriminator_test.sh " + "NONE " + self.path_test)
        return [float(line.strip().split(" ")[-1]) for line in open(self.path_cls_res)]

    def dev(self):
        os.system("cd " + self.path_discriminator + "; sh discriminator_test.sh " + self.path_AMT_dev_pos + " " + self.path_AMT_dev_neg)
        correct_num = 0; all_num = 0.0
        for line in open(self.path_cls_res):
            flist = line.strip().split(" ")
            if flist[1] == flist[2]:
                correct_num += 1
            all_num += 1
        return correct_num / all_num

    def test(self, log_no):
        os.system("cd " + self.path_discriminator + "; sh discriminator_test.sh " + self.path_AMT_test_pos + " " + self.path_AMT_test_neg + " " + str(log_no))
        correct_num = 0; all_num = 0.0
        for line in open(self.path_cls_res):
            flist = line.strip().split(" ")
            if flist[1] == flist[2]:
                correct_num += 1
            all_num += 1
        os.system("cd " + self.path_discriminator + "; sort -k5 -n test.res | awk \'{if($2==\"pos\")print 0; else print 1}\' > tmp")
        os.system("python evaluation_metrics.py > dis.res." + str(log_no))
        os.system("echo ACC " + str(correct_num / all_num) + " >> dis.res." + str(log_no))
        return correct_num / all_num

    def dev_epoch(self):
        os.system("cd " + self.path_discriminator + "; sh discriminator_test.sh " + self.file_dev_pos + " " + self.file_dev_neg)
        correct_num = 0; all_num = 0.0
        for line in open(self.path_cls_res):
            flist = line.strip().split(" ")
            if flist[1] == flist[2]:
                correct_num += 1
            all_num += 1
        return correct_num / all_num
