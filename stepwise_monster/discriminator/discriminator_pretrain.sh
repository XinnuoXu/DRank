#!/bin/bash

CLASSIFIER=hierarchical_coherence_attention
EXPERIMENT=experiments/dialogue_context_${CLASSIFIER}_classifier.json
MODEL=trained_models/${CLASSIFIER}

rm -fr $MODEL

allennlp train \
${EXPERIMENT} \
-s ${MODEL} \
--include-package coherence
#--include-package coherence > /dev/null

#--file-friendly-logging \

rm -rf trained_models.pretrain
cp -r trained_models trained_models.pretrain
