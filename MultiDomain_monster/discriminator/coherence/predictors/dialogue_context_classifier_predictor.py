from typing import Tuple

from allennlp.data import Instance
from allennlp.common.util import JsonDict
from overrides import overrides
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('dialogue_context_hierarchical_coherence_attention_predictor')
class DialogueContextClassifierPredictor(Predictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        context = json_dict['context']
        instance = self._dataset_reader.text_to_instance(context=context)

        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        all_labels = [label_dict[i] for i in range(len(label_dict))]

        return instance, {"all_labels": all_labels}
