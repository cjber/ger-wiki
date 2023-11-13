from collections import Counter
from zipfile import ZipFile

import pandas as pd
import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_lg")
wiki = pd.read_csv("data_processing/data/raw/wiki/wiki_info.csv").loc[:, "abs"].tolist()

ents = [
    ent.text
    for text in tqdm(nlp.pipe(wiki), total=len(wiki))
    for ent in text.ents
    if ent.label_ == "GPE"
]


ent_df = (
    pd.DataFrame.from_dict(Counter(ents), orient="index")
    .reset_index()
    .rename(columns={"index": "place", 0: "count"})
)

geonames = pd.read_csv(
    "./data_processing/data/raw/geonames/GB.txt",
    sep="\t",
    header=None,
    usecols=[1],
    names=["name"],
)

zip_file = ZipFile("data_processing/data/raw/geonames/opname_csv_gb.zip")
os = pd.concat(
    [
        pd.read_csv(
            zip_file.open(text_file.filename),
            low_memory=False,
            names=["osgb", "uri", "name"],
            usecols=["osgb", "uri", "name"],
        )
        for text_file in tqdm(zip_file.infolist())
        if text_file.filename.endswith(".csv")
    ]
)
geonames_unq = geonames["name"].append(os["name"])
ent_unq = ent_df["place"].unique()
new_places = set(ent_unq) - set(geonames_unq)

new_places = ent_df[ent_df["place"].isin(new_places)]
new_places.to_csv("data_processing/data/results/oog_spacy.csv", index=False)
