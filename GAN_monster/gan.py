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

TRAIN_LOG = open("./log/train.log", "w")

def train_generator_PG(gen, disc, gen_turns):

    # Sample from generator
    neg_examples = gen.sample(gen_turns, MAX_CONTEXT_LENGTH, "neg", GIVEN_TURNS, False)
    pos_examples = gen.sample(gen_turns, MAX_CONTEXT_LENGTH, "pos", GIVEN_TURNS, False)
    # Get rewards
    neg_rewards = disc.batchClassify(neg_examples)
    pos_rewards = disc.batchClassify(pos_examples)
    # Prepare data for seq2seq
    neg_gen_train_data, neg_rewards = gen.data_for_seq2seq(neg_examples, MAX_CONTEXT_LENGTH, GIVEN_TURNS, neg_rewards)
    pos_gen_train_data, pos_rewards = gen.data_for_seq2seq(pos_examples, MAX_CONTEXT_LENGTH, GIVEN_TURNS, pos_rewards)
    # Train generator
    gen_train_data = neg_gen_train_data + pos_gen_train_data
    rewards = neg_rewards + pos_rewards

    gen.train(gen_train_data, rewards)
    return neg_examples, neg_rewards

def train_discriminator(gen, disc, gen_turns):
    
    # Sample from generator
    neg_examples = gen.sample(gen_turns, MAX_CONTEXT_LENGTH, "neg", GIVEN_TURNS, False)
    pos_examples = gen.sample(gen_turns, MAX_CONTEXT_LENGTH, "pos", GIVEN_TURNS, False)
    # Train discriminator
    return disc.train(pos_examples, neg_examples)

def write_log(gen_len, epoch, examples, rewards):

    fpout = open("./log/generation_length_" + str(gen_len) + "_epoch_" + str(epoch) + ".log", "w")
    for i in range(0, len(examples)):
        ctx = examples[i].split("\t")
        fpout.write("\n".join(ctx) + "\n")
        fpout.write("Reward: " + str(rewards[i]) + "\n\n")
    fpout.close()

# MAIN
if __name__ == '__main__':

    gen = generator.Generator()
    disc = discriminator.Discriminator()

    '''
    # Pre-train generator on Maluba
    # gen.pre_train()

    # Pre-train generator on gold dials
    #gen.pre_train(max_ctx=MAX_CONTEXT_LENGTH, maluba=False)
    
    # Pre-train discriminator
    pos_examples = gen.sample(PRETRAIN_GEN_LENGTH, -1, "pos", GIVEN_TURNS, True)
    neg_examples = gen.sample(PRETRAIN_GEN_LENGTH, MAX_CONTEXT_LENGTH, "neg", GIVEN_TURNS, True)
    disc.pre_train(pos_examples, neg_examples, GIVEN_TURNS)

    # Report ACC on AMT-dev
    TRAIN_LOG.write('ACC of pretrained discriminator %f\n' % disc.dev())

    #ADV_TRAIN_EPOCHS = 3
    '''

    gen_len = 1; converged_tag = 0; epoch = 0

    while gen_len <= MAX_GEN_LENGTH:
        if epoch == 0:
            # LOGGING
            TRAIN_LOG.write('\n************ TRAINING FOR LENGTH-%d STARTS ************\n' % gen_len)
            examples = gen.sample(gen_len, MAX_CONTEXT_LENGTH, "neg", GIVEN_TURNS, False)
            rewards = disc.batchClassify(examples)
            write_log(gen_len, epoch, examples, rewards)
            gen.backup_model(epoch, gen_len)
            avg_rewards = sum(rewards)/len(rewards)
            TRAIN_LOG.write('[GENERATOR][G_length %d]Average reward at epoch %d is %f\n' % (gen_len, epoch, avg_rewards))

        epoch += 1

        # TRAIN GENERATOR
        examples, rewards = train_generator_PG(gen, disc, gen_len)
        avg_rewards = sum(rewards)/len(rewards)
        TRAIN_LOG.write('[GENERATOR][G_length %d]Average reward at epoch %d is %f\n' % (gen_len, epoch, avg_rewards))

        # LOGGING
        if epoch % REPORT_EPOCH == 0:
            write_log(gen_len, epoch, examples, rewards)
            gen.backup_model(epoch, gen_len)

        # TRAIN DISCRIMINATOR
        acc = train_discriminator(gen, disc, gen_len)
        if acc <= 0.65: converged_tag += 1
        TRAIN_LOG.write('[DISCRIMINATOR][G_length %d]ACC at epoch %d is %f\n' % (gen_len, epoch, acc))

        if converged_tag == 2:
            gen_len += 1; converged_tag = 0; epoch = 0
            disc.re_fresh()

