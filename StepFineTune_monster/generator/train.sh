# Start a new training round

batch_size=128
train_steps=50
report_steps=10
valid_steps=10
save_steps=10

#python train.py -data data/dialogue -save_model dialogue-model -train_steps $train_steps -report_every $report_steps -batch_size 256 -dropout 0.2 -src_word_vec_size 256 -tgt_word_vec_size 256 -rnn_size 128 -gpuid 0 -valid_steps $valid_steps -optim adam -learning_rate 0.001 -learning_rate_decay 1 -enc_layers 1 -dec_layers 1 -train_from dialogue-model.last/dialogue-model_best.pt -save_checkpoint_steps $save_steps

#Adam
python train.py -data data/dialogue -save_model dialogue-model -train_steps $train_steps -report_every $report_steps -batch_size $batch_size -dropout 0.2 -src_word_vec_size 256 -tgt_word_vec_size 256 -rnn_size 128 -gpuid 0 -valid_steps $valid_steps -optim adam -learning_rate 0.00001 -learning_rate_decay 1 -enc_layers 1 -dec_layers 1 -train_from dialogue-model.last/dialogue-model_best.pt -save_checkpoint_steps $save_steps
#SGD
#python train.py -data data/dialogue -save_model dialogue-model -train_steps $train_steps -report_every $report_steps -batch_size $batch_size -dropout 0.2 -src_word_vec_size 256 -tgt_word_vec_size 256 -rnn_size 128 -gpuid 0 -valid_steps $valid_steps -learning_rate 0.01 -enc_layers 1 -dec_layers 1 -train_from dialogue-model.last/dialogue-model_best.pt -save_checkpoint_steps $save_steps

# Move models for last training round
mv dialogue-model_step_$train_steps.pt dialogue-model_best.pt
rm -rf dialogue-model; mkdir dialogue-model; mv dialogue-model_* dialogue-model
rm -r dialogue-model.last
mv dialogue-model dialogue-model.last

