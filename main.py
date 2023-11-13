#!/usr/bin/env python

import torch
import typer
from allennlp.commands.train import train_model_from_file

from ger_wiki.batch_predictor import RunBatchPredictions

app = typer.Typer()


def train_model(name: str):
    typer.echo(f"Running {name}")
    try:
        train_model_from_file(
            parameter_filename=f"./configs/{name}.jsonnet",
            serialization_dir=f"./models/{name}_model",
            include_package=["ger_wiki", "allennlp_models"],
            force=True,
        )
    except FileNotFoundError as e:
        print(e)


cuda_device = 0 if torch.cuda.is_available() else -1


def get_predictions(name: str):
    # run predictions on Wikipedia corpus using second model
    batch_predictor = RunBatchPredictions(
        archive_path=f"./models/{name}_model/model.tar.gz",
        predictor_name="text_predictor",
        text_path="./data_processing/data/raw/wiki/wiki_info.csv",
        text_col="abs",
        cuda_device=cuda_device,
    )
    batch_predictor.run_batch_predictions(batch_size=8)
    batch_predictor.write_csv(csv_file="./data_processing/data/results/predictions.csv")
    batch_predictor.write_json(
        json_file="./data_processing/data/results/predictions.json"
    )


def main(
    name: str,
    predict: bool = False,
):
    """
    Choose an NER model to train, or use model predictor.

    :param name str: Config name (see configs/)\n
    :param baseline bool: Train baseline model\n
    :param predict bool: Label Wikipedia corpus using predictor\n
    """
    if predict:
        get_predictions(name)
    elif name == "all":
        for name in [
            "wiki_bert",
            "wiki_crf_basic",
            "wiki_crf",
            "wiki_distil",
            "wiki_roberta",
        ]:
            train_model(name)
    else:
        train_model(name)


if __name__ == "__main__":
    typer.run(main)
