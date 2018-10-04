import argparse
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
from coherence import *

parser = argparse.ArgumentParser()

parser.add_argument('-model', required=True, help='Path to model')

opts = parser.parse_args()


def main():
    archive = load_archive(opts.model)
    predictor = Predictor.from_archive(archive, 'dialogue_context_hierarchical_coherence_attention_predictor')

    context = response = ""

    test_set = []
    for l in open("coherence/dataset_readers/generated_dial_examples_test.pos"):
        test_set.append([l.strip(), "pos"])
    for l in open("coherence/dataset_readers/generated_dial_examples_test.neg"):
        test_set.append([l.strip(), "neg"])

    for pair in test_set:
        inputs = {"context": pair[0].split("\t")}
        result = predictor.predict_json(inputs)
        label = result.get("label")
        prob = max(result.get("class_probabilities"))
        #print("Predicted label: '{}' with probability: {}".format(label, prob))
        print ("RES", pair[1], label, prob)


if __name__ == "__main__":
    main()
