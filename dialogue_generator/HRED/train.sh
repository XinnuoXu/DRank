python train.py -data data/dialogue -save_model dialogue-model -train_steps 40000 -report_every 50 -batch_size 256 -dropout 0.2 -src_word_vec_size 128 -tgt_word_vec_size 128 -rnn_size 64 -gpuid 0 -copy_attn -sen_rnn_size 64 -valid_steps 50
#-save_checkpoint_steps 5 -valid_steps 5
