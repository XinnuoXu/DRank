import sys
import argparse
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
from coherence import *

parser = argparse.ArgumentParser()

parser.add_argument('-model', required=True, help='Path to model')
parser.add_argument('-test_pos', required=True, help='Path to test files (positive)')
parser.add_argument('-test_neg', required=True, help='Path to test files (negtive)')

opts = parser.parse_args()


def main():
    archive = load_archive(opts.model)
    predictor = Predictor.from_archive(archive, 'dialogue_context_hierarchical_coherence_attention_predictor')

    test_set = []
    if opts.test_pos != "NONE":
        for l in open(opts.test_pos):
            test_set.append([l.strip(), "pos"])
    if opts.test_neg != "NONE":
        for l in open(opts.test_neg):
            test_set.append([l.strip(), "neg"])

    for pair in test_set:
        inputs = {"context": pair[0].split("\t")}
        result = predictor.predict_json(inputs)
        print (result)
        label = result.get("label")
        prob = max(result.get("class_probabilities"))
        #print("Predicted label: '{}' with probability: {}".format(label, prob))
        print ("RES", pair[1], label, prob, result.get("class_probabilities")[1])


if __name__ == "__main__":
    main()
