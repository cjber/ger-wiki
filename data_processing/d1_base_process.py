"""
Create Wikipedia place dataset
"""
from data_processing.preprocess.spaceeval_conll import SpaceConll
from data_processing.preprocess.dbpedia_query import run_query, clean_abs
from pathlib import Path

DATA_PATH = Path("data_processing/data/")

space_conll = SpaceConll()
# output conll data
space_conll.output_conll(
    input_dir=DATA_PATH / "raw/spaceeval_data/", output_dir=DATA_PATH / "processed/"
)

wiki_csv = run_query()
wiki_csv = clean_abs(wiki_csv)

# sample to create pseudo labels from base model
wiki_csv.sample(n=200).to_csv(DATA_PATH / "interim/wiki/predict.csv")
wiki_csv.to_csv(DATA_PATH / "raw/wiki/wiki_info.csv")
