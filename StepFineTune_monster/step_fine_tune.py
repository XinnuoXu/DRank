from __future__ import print_function
from math import ceil
import numpy as np
import sys, os
import pdb

import torch
import torch.optim as optim
import torch.nn as nn

import generator
import discriminator

GIVEN_TURNS = 1
MAX_CONTEXT_LENGTH = 5
PRETRAIN_GEN_LENGTH = 10
MAX_GEN_LENGTH = 10
REPORT_EPOCH = 2
#ADV_TRAIN_EPOCHS = MAX_GEN_LENGTH * EPOCH_FOR_EACH_GEN_LENGTH

def train_generator_PG(gen, disc, gen_turns):

    # Sample from generator
    pos_examples = gen.sample(gen_turns, MAX_CONTEXT_LENGTH, "pos", GIVEN_TURNS, False)
    # Get rewards
    pos_rewards = [1] * len(pos_examples)
    # Prepare data for seq2seq
    gen_train_data, rewards = gen.data_for_seq2seq(pos_examples, MAX_CONTEXT_LENGTH, GIVEN_TURNS, pos_rewards)

    gen.train(gen_train_data, rewards)

# MAIN
if __name__ == '__main__':

    gen = generator.Generator()
    disc = discriminator.Discriminator()

    # Pre-train generator on Maluba
    gen.pre_train()

    gen_len = 1; converged_tag = 0; epoch = 0

    while gen_len <= MAX_GEN_LENGTH:
        train_generator_PG(gen, disc, gen_len)
        gen.backup_model(0, gen_len)
        gen_len += 1
