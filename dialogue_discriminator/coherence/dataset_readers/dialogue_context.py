import logging
from typing import Dict

import tqdm
from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, ListField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
import random
import re

from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("dialogue_context")
class DialogueContextDatasetReader(DatasetReader):

    def __init__(self, lazy: bool = False,
                 shuffle_examples: bool = True,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._shuffle_examples = shuffle_examples
        self._tokenizer = tokenizer or WordTokenizer(JustSpacesWordSplitter())
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        examples = self._read_file(file_path + ".pos", "pos") + self._read_file(file_path + ".neg", "neg")
        if self._shuffle_examples:
            random.shuffle(examples)
        for ex in examples:
            yield ex

    def _read_file(self, file_path, label):
        examples = []
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from file: %s", file_path)
            for line_num, line in enumerate(tqdm.tqdm(data_file.readlines())):
                line = line.strip("\n")
                dial = line.split("\t")
                examples.append(self.text_to_instance(dial, label))
        return examples

    @overrides
    def text_to_instance(self, context: str, label=None) -> Instance:  # type: ignore
        tmp = set()
        for i in range(0, len(context), 2):
            tmp.add(context[i])
        repeat = len(tmp) / ((len(context)+1)/2.0)
        tokenized_context = [self._tokenizer.tokenize(x) for x in context[:32]]
        context_field = ListField([TextField(t, self._token_indexers) for t in tokenized_context])

        fields = {'context': context_field, 'length': MetadataField(len(context)), 'repeat': MetadataField(repeat)}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'DialogueContextDatasetReader':
        lazy = params.pop('lazy', False)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        shuffle_examples = params.pop('shuffle_examples', False)
        params.assert_empty(cls.__name__)
        return cls(lazy=lazy, shuffle_examples=shuffle_examples,
                   tokenizer=tokenizer, token_indexers=token_indexers)
