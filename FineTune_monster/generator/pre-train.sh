# back-up training data
#cp -r data/ data.bakup

# preprocessing
#python preprocess.py -extra_vocab data/examples.dict -train_src data/train.en -train_tgt data/train.vi -valid_src data/dev.en -valid_tgt data/dev.vi -save_data data/dialogue -src_vocab_size 10000 -tgt_vocab_size 10000 -share_vocab -src_seq_length 150 -pre_train

# training
python train.py -data data/dialogue -save_model dialogue-model -train_steps 30001 -report_every 500 -batch_size 128 -dropout 0.2 -src_word_vec_size 256 -tgt_word_vec_size 256 -rnn_size 128 -gpuid 0 -valid_steps 5000 -optim adam -learning_rate 0.001 -learning_rate_decay 1 -enc_layers 1 -dec_layers 1 -save_checkpoint_steps 5000

# mv
mv dialogue-model_step_30000.pt dialogue-model_best.pt
rm -rf dialogue-model; mkdir dialogue-model; mv dialogue-model_* dialogue-model
mv dialogue-model dialogue-model.last
