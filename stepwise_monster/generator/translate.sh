python translate.py -model $1 -src data/test.en -tgt data/test.vi -report_bleu -verbose -gpu 0 -batch_size 128 -beam_size 1 -min_length 2 -max_length 25 -replace_unk -n_best 1 -beam_sample
