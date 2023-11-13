from zipfile import ZipFile

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


def read_wiki(input_file):
    wiki_place = pd.read_csv(
        input_file,
        header=None,
        names=["place", "type", "sentence", "wiki_title", "wiki_coords"],
    )

    wiki_place[["wiki_latitude", "wiki_longitude"]] = wiki_place[
        "wiki_coords"
    ].str.split(" ", 1, expand=True)
    wiki_place.drop("wiki_coords", axis=1)
    wiki_place = wiki_place[wiki_place["type"] == "PLACE_NAM"]
    return wiki_place


wiki_place = read_wiki("data_processing/data/results/predictions.csv")

geonames = pd.read_csv(
    "./data_processing/data/raw/geonames/GB.txt", sep="\t", header=None
).iloc[:, [0, 1, 4, 5]]
geonames.columns = ["id", "name", "lat", "lon"]
geonames = geonames[["name"]]

wiki_place_unq = wiki_place[["place"]].drop_duplicates()
new_places = set(wiki_place_unq["place"].tolist()) - set(geonames["name"].tolist())
new_places = wiki_place[wiki_place["place"].isin(new_places)]

new_counts = wiki_place["place"].value_counts().to_frame().reset_index()
new_counts = new_counts[new_counts["index"].isin(new_places["place"])]

new_counts.to_csv("./data_processing/data/results/distil_oog.csv")
