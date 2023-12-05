import random
from pathlib import Path

import jsonlines
import spacy
from spacy.training import biluo_tags_to_offsets, offsets_to_biluo_tags


def split_text(
    input_file: str, large_file: str, sample_file: str, sample: int, seed: int
):
    """
    Split a larger file into two by line by a proportion of the sample.

    Converts files to sets, removing duplicates before splitting.

    :param input_file str: Input text file, line separated.
    :param large_file str: Output for larger split.
    :param sample_file str: Output for smaller split.
    :param sample int: Proportion to split large file by.
    """
    random.seed(seed)

    text = set(open(input_file).readlines())
    subset = set(random.sample(text, sample))
    large_subset = text - subset

    open(large_file, "w").write("".join(list(large_subset)))
    open(sample_file, "w").write("".join(list(subset)))


def predictions_to_doccano(input_file: Path, output_file: Path):
    """
    Convert AllenNLP output json to Doccano style.

    :param input_file str: Input AllenNLP json file.
    :param output_file str: Doccano style output json.
    """
    nlp = spacy.blank("en")
    json_lines = []
    with jsonlines.open(input_file) as reader:
        for obj in reader:
            text = obj["sentence"]
            doc = nlp(text)
            offsets = biluo_tags_to_offsets(doc, obj["tags"])

            json_line = {"text": text, "labels": offsets}
            json_lines.append(json_line)

    with jsonlines.open(output_file, mode="w") as writer:
        for line in json_lines:
            writer.write(line)


def doccano_to_conll(input_file: Path, output_file: Path, language="en"):
    """
    Convert Doccano json output to CoNLL scheme.

    :param input_file str: Input Doccano json.
    :param output_file str: Output CoNLL file.
    :param language Spacy language: For tokenization.
    """
    nlp = spacy.blank(language)
    doc_lines = []
    with jsonlines.open(input_file) as reader:
        for obj in reader:
            doc = nlp(obj["text"])
            tokens = [str(token) for token in doc]
            tags = offsets_to_biluo_tags(doc, obj["labels"])
            doc_zip = list(zip(tokens, tags))
            doc_zip.append(("", "\n"))
            doc_lines.extend(doc_zip)
            doc_lines = [list(pair) for pair in doc_lines]
            doc_lines = [pair for pair in doc_lines if pair[0] != " "]

        doc_lines = [pair for pair in doc_lines if pair[1] != "-"]

    with open(output_file, "w") as fp:
        fp.write("\n".join(f"{x[0]} {x[1]}" for x in doc_lines))


if __name__ == "__main__":
    data_path = Path("./data_processing/data/")

    # convert prediction outputs into doccano input
    predictions_to_doccano(
        input_file=data_path / "interim/wiki/predictions.jsonl",
        output_file=data_path / "interim/wiki/doccano_input.jsonl",
    )

    # split labelled doccano data for training and evaluation
    split_text(
        input_file=data_path / "interim/wiki/doccano/doccano_output.json1",
        large_file=data_path / "interim/wiki/wiki_train.jsonl",
        sample_file=data_path / "interim/wiki/wiki_test.jsonl",
        sample=20,
    )

    doccano_to_conll(
        input_file=data_path / "interim/wiki/wiki_train.jsonl",
        output_file=data_path / "processed/wiki_train.conll",
    )
    doccano_to_conll(
        input_file=data_path / "interim/wiki/wiki_test.jsonl",
        output_file=data_path / "processed/wiki_test.conll",
    )
