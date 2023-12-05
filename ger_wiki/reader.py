import itertools
from typing import Any, Dict, Iterator, List

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    Field,
    MetadataField,
    SequenceLabelField,
    TextField
)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp_models.tagging.models import CrfTagger  # noqa: F401
from overrides import overrides


def _is_divider(line: str) -> bool:
    return not line.strip()


@DatasetReader.register("ger_reader")
class GerReader(DatasetReader):
    def __init__(
        self, token_indexers: Dict[str, TokenIndexer] = None, lazy: bool = False
    ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, "r") as conll_file:
            for divider, lines in itertools.groupby(conll_file, _is_divider):
                if divider:
                    continue
                fields = [line.strip().split() for line in lines]
                fields = list(zip(*fields))
                tokens, ner_tags = fields
                ner_tags = ["O" if tag[-3:] == "NOM" else tag for tag in ner_tags]
                yield self.text_to_instance(tokens, ner_tags)

    @overrides
    def text_to_instance(
        self,
        words: List[str],
        ner_tags: List[str] = None,
        sentence: str = None,
        place: str = None,
        wiki_point: str = None,
    ) -> Instance:
        metadata_dict: Dict[str, Any] = {}
        fields: Dict[str, Field] = {}

        text_field = TextField([Token(w) for w in words], self._token_indexers)

        # only include tags if present
        if ner_tags:
            metadata_dict["gold_tags"] = ner_tags
            fields["tags"] = SequenceLabelField(ner_tags, text_field)

        # only include wiki metadata if present
        if sentence:
            metadata_dict["sentence"] = sentence
        if place:
            metadata_dict["place"] = place
        if wiki_point:
            metadata_dict["wiki_point"] = wiki_point

        metadata_dict["words"] = words

        fields["tokens"] = text_field
        fields["metadata"] = MetadataField(metadata_dict)
        return Instance(fields)
