import torch
import random
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
import torch.nn.init as init
import os, sys
import tokenize
import io
from pydial import DialogueManager, DialogueGenerator

class Generator(nn.Module):

    def __init__(self):
        self.gold_dials = [[self.preprocess(item) for item in ex.strip().split("\t")] for ex in open("gold_dialogues.in")]
        self.path_generator = "./generator/"
        self.path_train_en = "data/train.en"
        self.path_train_vi = "data/train.vi"
        self.path_dev_en = "data/dev.en"
        self.path_dev_vi = "data/dev.vi"
        self.path_test_en = "data/test.en"
        self.path_test_vi = "data/test.vi"
        self.path_pred = "pred.txt"
        self.train_dev_rate = 0.8

    def preprocess(self, line):
        f = io.StringIO(line.lower())
        tok_list = []
        for tok in tokenize.generate_tokens(f.readline):
            _, tok_str, _, _, _ = tok
            tok_list.append(tok_str)
        return " ".join(tok_list).strip()
    
    def get_sys_utt(self, context, user_utt, pre_train):
        dg = DialogueGenerator()
        sys_utt = dg.first_system_act()
        u_utt_set = set()
        restaurant_name = ""
        for i in range(1, len(context), 2):
            u_utt = context[i]
            u_utt_set.add(u_utt)
            sys_utt, restaurant_name = dg.next_system_act(u_utt)
        if (user_utt.find("phone number") > -1 or user_utt.find("address") > -1) and restaurant_name != "" and restaurant_name != "none":
            user_utt += " " + restaurant_name
        sys_utt, restaurant_name = dg.next_system_act(user_utt)
        sys_utt = self.preprocess(sys_utt)
        u_utt_set.add(user_utt)
        dg.end_call()
        if sys_utt.strip() == "":
            return "", 1
        if pre_train and len(context) >=5  and user_utt == context[-2] and user_utt == context[-4]:
            return sys_utt, 1
        return sys_utt, 0

    def sample(self, gen_turns, max_ctx, tag, given_turns, pre_train, model_path = ""):
        if model_path == "":
            model_path = "dialogue-model.last"

        if tag == "pos":
            # first turn is given
            examples = ["\t".join(dial[:min(given_turns + gen_turns * 2, len(dial))]) for dial in self.gold_dials]
            return examples

        # sample pre-generated dialogues
        contexts = [dial[:given_turns] for dial in self.gold_dials]
        end_signal = np.zeros(len(contexts))

        # given context generate the rest of dialogue
        while gen_turns > 0:
            gen_turns -= 1

            # generate user_utt
            fp_test_en = open(self.path_generator + self.path_test_en, "w")
            fp_test_vi = open(self.path_generator + self.path_test_vi, "w")
            for context in contexts:
                fp_test_en.write(" <s> ".join(context[max(0, len(context) - max_ctx):]) + "\n")
                fp_test_vi.write(context[-1] + "\n")
            fp_test_en.close(); fp_test_vi.close()
            os.system("cd " + self.path_generator + "; sh translate.sh " + model_path)
            gen_uttrs = [line.strip() for line in open(self.path_generator + self.path_pred)]

            # get system responses from dail_systems
            for i in range(0, len(contexts)):
                if end_signal[i]:
                    continue
                sys_utt, end_signal[i] = self.get_sys_utt(contexts[i], gen_uttrs[i], pre_train)
                if not end_signal[i]:
                    contexts[i].append(gen_uttrs[i])
                    contexts[i].append(sys_utt)
            if sum(end_signal) == len(end_signal):
                break
        return ["\t".join(ctx) for ctx in contexts]

    def data_for_seq2seq_v1(self, contexts, max_ctx, given_turns):
        # generate training data for generator
        for_gen = []
        for i in range(0, len(contexts)):
            context = contexts[i].split("\t")
            for_gen.append(" <s> ".join(context[max(0, len(context) - 2 - max_ctx):-2]) + "\t" + context[-2])
        return for_gen

    def data_for_seq2seq(self, ctxs, max_ctx, given_turns, rewards):
        contexts = []; new_rewards = []
        for i in range(0, len(ctxs)):
            dial = ctxs[i].split("\t")
            for j in range(given_turns + 2, len(dial) + 1, 2):
                new_rewards.append(rewards[i])
                contexts.append(dial[:j])
        for_gen = []
        for context in contexts:
            for_gen.append(" <s> ".join(context[max(0, len(context) - 2 - max_ctx):-2]) + "\t" + context[-2])
        return for_gen, new_rewards

    def batch_norm(self, rewards):
        mean = np.mean(rewards)
        var = np.var(rewards)
        #normalize = (rewards - mean) / (var ** 0.5 + 0.005) + 1
        normalize = rewards - mean + 1
        return normalize
        
    def train(self, examples, rewards):
        train_pos = int(self.train_dev_rate * len(examples))
        fp_train_en = open(self.path_generator + self.path_train_en, "w")
        fp_train_vi = open(self.path_generator + self.path_train_vi, "w")
        rewards = self.batch_norm(rewards)

        for i in range (0, train_pos):
            item = examples[i]
            ctx = item.split("\t")[0]; rep = item.split("\t")[1]
            fp_train_en.write(str(rewards[i]) + " " + ctx + "\n")
            fp_train_vi.write(rep + "\n")
        fp_train_en.close(); fp_train_vi.close()

        fp_dev_en = open(self.path_generator + self.path_dev_en, "w")
        fp_dev_vi = open(self.path_generator + self.path_dev_vi, "w")
        for i in range(train_pos, len(examples)):
            item = examples[i]
            ctx = item.split("\t")[0]; rep = item.split("\t")[1]
            fp_dev_en.write(ctx + "\n")
            fp_dev_vi.write(rep + "\n")
        fp_dev_en.close(); fp_dev_vi.close()

        os.system("cd " + self.path_generator + "; rm data/dialogue.train* data/dialogue.valid*; sh preprocess.sh; sh train.sh")

    def backup_model(self, epoch, gen_len):
        backup_file = "dialogue-model.epoch" + str(epoch) + ".gen" + str(gen_len)
        os.system("cd " + self.path_generator + "; cp -r dialogue-model.last " + backup_file)

    def pre_train(self, max_ctx=-1, maluba=True):
        if maluba:
            os.system("cd " + self.path_generator + "; sh pre-train.sh")
        else:
            train_pos = int(self.train_dev_rate * len(self.gold_dials))
            fp_train_en = open(self.path_generator + self.path_train_en, "w")
            fp_train_vi = open(self.path_generator + self.path_train_vi, "w")
            for ex in self.gold_dials[:train_pos]:
                for i in range(1, len(ex), 2):
                    start_pos = max(0, i-max_ctx)
                    while start_pos < i:
                        context = " <s> ".join(ex[start_pos:i])
                        response = ex[i]
                        fp_train_en.write("1 " + context + "\n")
                        fp_train_vi.write(response + "\n")
                        start_pos += 2
            fp_train_en.close(); fp_train_vi.close()

            fp_dev_en = open(self.path_generator + self.path_dev_en, "w")
            fp_dev_vi = open(self.path_generator + self.path_dev_vi, "w")
            for ex in self.gold_dials[train_pos:]:
                for i in range(1, len(ex), 2):
                    start_pos = max(0, i-max_ctx)
                    while start_pos < i:
                        context = " <s> ".join(ex[max(0, i-max_ctx):i])
                        response = ex[i]
                        fp_dev_en.write("1 " + context + "\n")
                        fp_dev_vi.write(response + "\n")
                        start_pos += 2
            fp_dev_en.close(); fp_dev_vi.close()
            os.system("cd " + self.path_generator + "; rm data/dialogue.train* data/dialogue.valid*; sh preprocess.sh; sh train.sh")
                    

if __name__ == '__main__':
    MAX_GEN_LENGTH = 11
    MAX_CONTEXT_LENGTH = 5
    GIVEN_TURNS = 1

    gen = Generator()
    contexts = gen.sample(MAX_GEN_LENGTH, MAX_CONTEXT_LENGTH, "neg", GIVEN_TURNS, False, model_path = sys.argv[1])

    fpout = open(gen.path_generator + sys.argv[1] + "." + sys.argv[2] + ".ex", "w")
    for item in contexts:
        fpout.write(item + "\n")
    fpout.close()
