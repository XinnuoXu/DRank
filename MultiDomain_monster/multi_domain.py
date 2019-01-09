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

if __name__ == '__main__':

    gen = generator.Generator()

    # Pre-train generator on Maluba
    gen.pre_train()
