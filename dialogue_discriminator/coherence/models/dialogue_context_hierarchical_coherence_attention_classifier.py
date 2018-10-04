from typing import Optional, Dict

import numpy
import torch
import torch.nn
import torch.nn.functional as F
from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2VecEncoder, \
    TextFieldEmbedder, Seq2SeqEncoder, SimilarityFunction, \
    TimeDistributed, MatrixAttention
from allennlp.nn import InitializerApplicator, RegularizerApplicator

from allennlp.nn.util import get_text_field_mask, last_dim_softmax, weighted_sum
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("dialogue_context_hierarchical_coherence_attention_classifier")
class DialogueContextHierarchicalCoherenceAttentionClassifier(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 attend_feedforward: FeedForward,
                 matrix_attention: MatrixAttention,
                 compare_feedforward: FeedForward,
                 classifier_feedforward: FeedForward,
                 final_classifier_feedforward: FeedForward,
                 utterance_encoder: Seq2VecEncoder,
                 context_encoder: Seq2SeqEncoder,
                 response_encoder: Optional[Seq2SeqEncoder] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(DialogueContextHierarchicalCoherenceAttentionClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = vocab.get_vocab_size("labels")
        self.utterances_encoder = TimeDistributed(utterance_encoder)
        self.context_encoder = context_encoder
        self.response_encoder = response_encoder
        self.attend_feedforward = TimeDistributed(attend_feedforward)
        self.matrix_attention = matrix_attention
        self.compare_feedforward = TimeDistributed(compare_feedforward)
        self.classifier_feedforward = classifier_feedforward
        self.final_classifier_feedforward = final_classifier_feedforward
        labels = self.vocab.get_index_to_token_vocabulary('labels')
        pos_label_index = list(labels.keys())[list(labels.values()).index('neg')]

        check_dimensions_match(text_field_embedder.get_output_dim(), attend_feedforward.get_input_dim(),
                               "text field embedding dim", "attend feedforward input dim")
        check_dimensions_match(classifier_feedforward.get_output_dim(), self.num_classes,
                               "final output dimension", "number of labels")

        self.metrics = {
            "accuracy": CategoricalAccuracy()
            # "f1": F1Measure(positive_label=pos_label_index)
        }
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    @overrides
    def forward(self,
                context: Dict[str, torch.LongTensor],
                length: torch.LongTensor = None,
                repeat: torch.FloatTensor = None,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:

        expected_dim = self.final_classifier_feedforward.get_input_dim() / 2
        dia_len = context['tokens'].size()[1]
        if expected_dim - dia_len > 0:
            padding = torch.zeros([context['tokens'].size()[0], (expected_dim - dia_len), context['tokens'].size()[2]]).long()
            context['tokens'] = torch.cat([context['tokens'], padding], dim=1)

        # context: batch_size * dials_len * sentences_len
        # embedded_context: batch_size * dials_len * sentences_len * emb_dim
        embedded_context = self.text_field_embedder(context)
        # utterances_mask: batch_size * dials_len * sentences_len
        utterances_mask = get_text_field_mask(context, 1).float()
        # encoded_utterances: batch_size * dials_len * emb_dim
        encoded_utterances = self.utterances_encoder(embedded_context, utterances_mask)
        # embedded_context: batch_size * (dials_len - 1) * emb_dim
        embedded_context = encoded_utterances[:, :-1, :]

        # embedded_response: batch_size * (dials_len - 1) * emb_dim
        embedded_response = encoded_utterances[:, 1:, :]
        # response_mask: batch_size * (dials_len - 1)
        response_mask = get_text_field_mask(context).float()[:, 1:]
        # context_mask: batch_size * (dials_len - 1)
        context_mask = get_text_field_mask(context).float()[:, :-1]

        projected_context = self.attend_feedforward(embedded_context)
        projected_response = self.attend_feedforward(embedded_response)

        # similarity_matrix: batch_size * (dials_len - 1) * (dials_len - 1)
        similarity_matrix = self.matrix_attention(projected_context, projected_response)

        # c2r_attention: batch * (dials_len - 1) * (dials_len - 1)
        c2r_attention = last_dim_softmax(similarity_matrix, response_mask)
        # attended_response: batch * (dials_len - 1) * emb_dim
        attended_response = weighted_sum(embedded_response, c2r_attention)

        # r2c_attention: batch * (dials_len - 1) * (dials_len - 1)
        r2c_attention = last_dim_softmax(similarity_matrix.transpose(1, 2).contiguous(), context_mask)
        # attended_context: batch * (dials_len - 1) * emb_dim
        attended_context = weighted_sum(embedded_context, r2c_attention)


        # context_compare_input: batch * (dials_len - 1) * (emb_dim + emb_dim)
        context_compare_input = torch.cat([embedded_context, attended_response], dim=-1)
        # response_compare_input: batch * (dials_len - 1) * (emb_dim + emb_dim)
        response_compare_input = torch.cat([embedded_response, attended_context], dim=-1)

        # compared_context: batch * (dials_len - 1) * emb_dim
        compared_context = self.compare_feedforward(context_compare_input)
        compared_context = compared_context * context_mask.unsqueeze(-1)

        # compared_response: batch * (dials_len - 1) * emb_dim
        compared_response = self.compare_feedforward(response_compare_input)
        compared_response = compared_response * response_mask.unsqueeze(-1)

        # aggregate_input: batch * (dials_len - 1) * (compare_context_dim + compared_response_dim)
        aggregate_input = torch.cat([compared_context, compared_response], dim=-1)

        # class_logits & class_probs:  batch * (dials_len - 1) * 2
        class_logits = self.classifier_feedforward(aggregate_input)
        class_probs = F.softmax(class_logits, dim=-1).reshape(class_logits.size()[0], -1)
        length_tensor = torch.FloatTensor(length).reshape(-1, 1)
        repeat_tensor = torch.FloatTensor(repeat).reshape(-1, 1)
        class_probs = torch.cat([class_probs, length_tensor, repeat_tensor], dim=1)

        full_logits = self.final_classifier_feedforward(class_probs)
        full_probs = F.softmax(full_logits, dim=-1)
        output_dict = {"class_logits": full_logits, "class_probabilities": full_probs}

        if label is not None:
            loss = self.loss(full_logits, label.squeeze(-1))
            for metric in self.metrics.values():
                metric(full_logits, label.squeeze(-1))
            output_dict['loss'] = loss

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # precision, recall, f1 = self.metrics["f1"].get_metric(reset)
        # metrics = {"accuracy": self.metrics["accuracy"].get_metric(reset),
        #            "precision:": precision,
        #            "recall": recall,
        #            "f1": f1}
        # return metrics
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predictions = output_dict["class_probabilities"].cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels") for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'DialogueContextHierarchicalCoherenceAttentionClassifier':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)

        utterance_encoder = Seq2VecEncoder.from_params(params.pop("utterance_encoder"))
        context_encoder = Seq2SeqEncoder.from_params(params.pop("context_encoder"))

        response_encoder_params = params.pop("response_encoder", None)
        if response_encoder_params is not None:
            response_encoder = Seq2SeqEncoder.from_params(response_encoder_params)
        else:
            response_encoder = None

        attend_feedforward = FeedForward.from_params(params.pop('attend_feedforward'))
        #similarity_function = SimilarityFunction.from_params(params.pop("similarity_function"))
        compare_feedforward = FeedForward.from_params(params.pop('compare_feedforward'))
        classifier_feedforward = FeedForward.from_params(params.pop("classifier_feedforward"))
        final_classifier_feedforward = FeedForward.from_params(params.pop("final_classifier_feedforward"))

        initializer = InitializerApplicator.from_params(params.pop("initializer", []))
        regularizer = RegularizerApplicator.from_params(params.pop("regularizer", []))

        matrix_attention = MatrixAttention().from_params(params.pop("similarity_function"))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   attend_feedforward=attend_feedforward,
                   matrix_attention=matrix_attention,
                   compare_feedforward=compare_feedforward,
                   classifier_feedforward=classifier_feedforward,
                   final_classifier_feedforward=final_classifier_feedforward,
                   utterance_encoder=utterance_encoder,
                   context_encoder=context_encoder,
                   response_encoder=response_encoder,
                   initializer=initializer,
                   regularizer=regularizer)
