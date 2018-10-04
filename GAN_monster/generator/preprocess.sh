rm data/dialogue.train*.pt data/dialogue.valid*.pt
python preprocess.py -extra_vocab data/examples.dict -train_src data/train.en -train_tgt data/train.vi -valid_src data/dev.en -valid_tgt data/dev.vi -save_data data/dialogue -src_vocab_size 15000 -tgt_vocab_size 10000 -share_vocab -src_seq_length 150
