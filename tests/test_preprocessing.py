import warnings
from pathlib import Path

import jsonlines

from data_processing.preprocess import doccano_functions as df

DIR = Path("tests/fixtures")


class TestDoccanoFunctions:
    def test_split_text(self):
        input_file = DIR / "split_text/toy_split.txt"
        large_file = DIR / "split_text/toy_large.txt"
        sample_file = DIR / "split_text/toy_sample.txt"
        sample = 2
        seed = 42

        df.split_text(
            input_file=input_file,
            large_file=large_file,
            sample_file=sample_file,
            sample=sample,
            seed=seed,
        )

        toy_data_length = len(open(input_file).readlines())
        large_file_length = len(open(large_file).readlines())
        sample_file_length = len(open(sample_file).readlines())

        assert large_file_length == toy_data_length - sample_file_length
        assert large_file_length == toy_data_length - sample
        assert sample_file_length == sample

    def test_predictions_to_doccano(self):
        input_file = DIR / "predictions_to_doccano/toy_predictions.jsonl"
        output_file = DIR / "predictions_to_doccano/toy_doccano.jsonl"

        df.predictions_to_doccano(input_file=input_file, output_file=output_file)

        doccano_list = []
        with jsonlines.open(output_file) as reader:
            for obj in reader:
                doccano_list.append(obj)

        assert doccano_list[0] == {
            "text": "Blairgowrie and Rattray  is a twin burgh in Perth and Kinross, Scotland.",
            "labels": [
                [0, 11, "PLACE_NAM"],
                [16, 23, "PLACE_NAM"],
                [30, 34, "PLACE_NOM"],
                [41, 43, "PLACE_NAM"],
                [50, 62, "PLACE_NAM"],
            ],
        }
        assert doccano_list[1] == {
            "text": "Locals refer to the town as Blair.",
            "labels": [[20, 24, "PLACE_NOM"], [28, 33, "PLACE_NAM"]],
        }

    def test_doccano_to_conll(self):
        """
        One instance with no misaligned tags, and one with misalignments.
        """
        input_file = DIR / "doccano_to_conll/toy_doccano.jsonl"
        output_file = DIR / "doccano_to_conll/toy_conll.conll"

        with warnings.catch_warnings(record=True) as w:
            df.doccano_to_conll(input_file=input_file, output_file=output_file)
            # test the warning is due to misalignment
            assert issubclass(w[-1].category, UserWarning)

        read_output = open(output_file, "r").read().splitlines()
        conll_correct = [
            "No O",
            "separate O",
            "population O",
            "statistic O",
            "is O",
            "available O",
            "for O",
            "Hop B-PLACE_NAM",
            "Pole L-PLACE_NAM",
            ". O",
            " ",
            "",
            "No O",
            "separate O",
            "population O",
            "statistic O",
            "is O",
            "available O",
            "for O",
            ". O",
            " ",
        ]

        assert read_output == conll_correct
