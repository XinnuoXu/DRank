#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import argparse

from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator

import onmt.inputters
import onmt.translate
import onmt
import onmt.model_builder
import onmt.modules
import onmt.opts

from tokenize import generate_tokens
import io


def main(opt):
    translator = build_translator(opt, report_score=True)
    conversation = []
    while 1:
        # Read line
        line = input("SYS: ").strip()
        if line == "clean":
            del conversation[:]
            continue
        # Preprocess input
        tokens = []
        for tok in generate_tokens(io.StringIO(line.lower()).readline):
            _, t_str, _, _, _ = tok
            tokens.append(t_str)
        # Put input into tmp file
        if len(conversation) == 5:
            del conversation[0]
        conversation.append(" ".join(tokens))
        fpin = open("tmp_src", "w")
        fpin.write(" <s> ".join(conversation) + "\n")
        fpin.close()
        # Get response
        translator.translate(src_path="tmp_src",
                         tgt_path="tmp_tgt",
                         src_dir="./",
                         batch_size=1,
                         attn_debug=opt.attn_debug)
        # Read result
        fpout = open("pred.txt")
        user_utt = fpout.readlines()[-1].strip()
        fpout.close()
        if len(conversation) == 5:
            del conversation[0]
        conversation.append(user_utt)
        print ("\nUSER: ", user_utt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)

    opt = parser.parse_args()
    logger = init_logger(opt.log_file)
    main(opt)
