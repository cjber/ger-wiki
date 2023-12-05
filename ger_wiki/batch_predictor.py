import os
import re

import jsonlines
import pandas as pd
import spacy
from allennlp.data.dataset_readers.dataset_utils import bioul_tags_to_spans
from allennlp.models.archival import load_archive
from tqdm import tqdm

from ger_wiki.predictor import TextPredictor


class RunBatchPredictions:
    def __init__(
        self,
        archive_path: str,
        predictor_name: str,
        text_path: str,
        text_col: str,
        cuda_device: int,
        language: str = "en_core_web_sm",
    ):
        archive = load_archive(archive_path, cuda_device=cuda_device)
        self.predictor = TextPredictor.from_archive(
            archive, predictor_name=predictor_name
        )

        self._nlp = spacy.load(language)
        self._nlp.add_pipe("sentencizer")

        self.text = self.read_lines(text_path, text_col)

    def read_lines(self, text_path, text_col):
        csv: pd.DataFrame = pd.read_csv(text_path, index_col=0)
        # very simple sentencizer (spacy far too slow)
        csv[text_col] = csv[text_col].str.split("\. ")
        csv = csv.explode(text_col)
        csv[text_col] = csv[text_col].astype(str)
        # remove very short sentences that are likely incomplete
        csv = csv[csv[text_col].apply(lambda x: len(x) > 10)]

        return [
            {
                "sentence": row[text_col],
                "place": row["label"],
                "wiki_point": row["point"],
            }
            for _, row in csv.iterrows()
        ]

    def run_batch_predictions(self, batch_size):
        chunks = (len(self.text) - 1) // batch_size + 1
        self.predictions = []
        for i in tqdm(range(chunks)):
            batches = self.text[i * batch_size : (i + 1) * batch_size]
            batches_out = self.predictor.predict_batch_json(batches)
            self.predictions.extend(iter(batches_out))

    def write_json(self, json_file):
        if os.path.exists(json_file):
            os.remove(json_file)

        with jsonlines.open(json_file, mode="w") as writer:
            for line in self.predictions:
                writer.write(line)

    def write_csv(self, csv_file):
        if os.path.exists(csv_file):
            os.remove(csv_file)

        for batch in self.predictions:
            words = batch["words"]
            spans = bioul_tags_to_spans(batch["tags"])
            tags_list = []

            for span in spans:
                offsets = span[1]
                label = span[0]
                word = " ".join(words[offsets[0] : offsets[1] + 1])
                # fix apostrophes and dashes
                word = re.sub(r"\s([\'])", r"\1", word)
                word = re.sub(r"\s([-])\s", r"\1", word)

                tags_list.append(
                    (
                        word,
                        label,
                        batch["sentence"],
                        batch["place"],
                        batch["wiki_point"],
                    )
                )

            if tags_list != []:
                tags_dataframe = pd.DataFrame(tags_list)
                tags_dataframe.columns = [
                    "Place",
                    "Type",
                    "Sentence",
                    "Place",
                    "wiki_point",
                ]
                tags_dataframe.to_csv(csv_file, mode="a", header=False, index=False)
