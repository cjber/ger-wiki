from typing import Dict, List

import spacy
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers import SpacyTokenizer
from allennlp.models import Model
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.predictors.predictor import Predictor
from overrides import overrides
from spacy.tokens import Doc


@Predictor.register("text_predictor")
class TextPredictor(SentenceTaggerPredictor):
    """
    Predictor that inherits the sentence predictor.

    Processes strings to sentences and outputs a list of tags with additional
    metadata. This predictor uses spacy sentencizer to split long strings into
    sentences.
    BIOUL tags handled by predictions_to_labeled_instances (inherited).
    """

    def __init__(
        self,
        model: Model,
        dataset_reader: DatasetReader,
        language: str = "en_core_web_sm",
    ) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyTokenizer(language=language, pos_tags=True)
        self._nlp = spacy.load(language)

        self._nlp.add_pipe("sentencizer")

    @overrides
    def load_line(self, line: str) -> JsonDict:
        """
        Processes strings into sentences from text, rather than JSON-lines.
        """
        doc: Doc = self._nlp(line)
        sentences: List[str] = [sent.text for sent in doc.sents]

        for sent in sentences:
            return {"sentence": sent}

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"sentence": "..."}`.
        Runs the underlying model, and adds the `"words"` to the output.
        """
        sentence = json_dict["sentence"]
        if "place" in json_dict:
            place = json_dict["place"]
        if "wiki_point" in json_dict:
            wiki_point = json_dict["wiki_point"]

        tokens = [str(token) for token in self._tokenizer.tokenize(sentence)]

        try:
            return self._dataset_reader.text_to_instance(
                tokens, sentence=sentence, place=place, wiki_point=wiki_point
            )
        except NameError:
            return self._dataset_reader.text_to_instance(tokens)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        """
        Runs model over instances.
        Model outputs logits, mask and tags by default. This adds the
        wiki metadata.
        """
        output: Dict = self._model.forward_on_instance(instance)

        if "sentence" in output:
            output["sentence"] = instance.fields["metadata"]["sentence"]
        if "place" in output:
            output["place"] = instance.fields["metadata"]["place"]
        if "place" in output:
            output["wiki_point"] = instance.fields["metadata"]["wiki_point"]

        return sanitize(output)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> JsonDict:
        """
        Runs model over batches of instances.
        Adds metadata as with predict_instance.
        """
        outputs = self._model.forward_on_instances(instances)
        sentences = iter(
            [instance.fields["metadata"]["sentence"] for instance in instances]
        )
        places = iter([instance.fields["metadata"]["place"] for instance in instances])
        wiki_points = iter(
            [instance.fields["metadata"]["wiki_point"] for instance in instances]
        )

        for output in outputs:
            output["sentence"] = next(sentences)
            output["place"] = next(places)
            output["wiki_point"] = next(wiki_points)

        return sanitize(outputs)
