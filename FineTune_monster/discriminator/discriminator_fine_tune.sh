#!/bin/bash

rm -rf trained_models.last
mv trained_models trained_models.last

CLASSIFIER=hierarchical_coherence_attention
EXPERIMENT=experiments/dialogue_context_${CLASSIFIER}_classifier.json
MODEL=trained_models.last/${CLASSIFIER}/model.tar.gz
SERIALIZATION=trained_models/${CLASSIFIER}

allennlp fine-tune -c ${EXPERIMENT} -m ${MODEL} -s ${SERIALIZATION} --include-package coherence

cp ./trained_models.last/neg.examples  ./trained_models
