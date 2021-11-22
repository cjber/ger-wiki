"""
This section takes manually labelled entities from Doccano and converts into
training and testing CoNLL formatted data for the second model
"""
from data_processing.preprocess import doccano_functions as dcf
from pathlib import Path

data_path = Path("./data_processing/data/")

# convert prediction outputs into doccano input
dcf.predictions_to_doccano(
    input_file=data_path / "interim/wiki/predictions.jsonl",
    output_file=data_path / "interim/wiki/doccano_input.jsonl",
)

# split labelled doccano data for training and evaluation
dcf.split_text(
    input_file=data_path / "interim/wiki/doccano/doccano_output.json1",
    large_file=data_path / "interim/wiki/wiki_train.jsonl",
    sample_file=data_path / "interim/wiki/wiki_test.jsonl",
    sample=20,
)

dcf.doccano_to_conll(
    input_file=data_path / "interim/wiki/wiki_train.jsonl",
    output_file=data_path / "processed/wiki_train.conll",
)
dcf.doccano_to_conll(
    input_file=data_path / "interim/wiki/wiki_test.jsonl",
    output_file=data_path / "processed/wiki_test.conll",
)
