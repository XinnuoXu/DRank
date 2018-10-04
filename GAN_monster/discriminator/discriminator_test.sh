#python predictor_coherence_interactive.py -model trained_models/hierarchical_coherence_attention/model.tar.gz | grep RES | awk '{if($2==$3)c+=1; else f+=1}END{print c/(c+f)}' > test.res
python predictor_coherence_interactive.py -model trained_models/hierarchical_coherence_attention/model.tar.gz -test_pos $1 -test_neg $2 | grep RES > test.res
