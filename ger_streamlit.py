import re

from allennlp.data.dataset_readers.dataset_utils import bioul_tags_to_spans
from allennlp.models.archival import load_archive
import streamlit as st
import torch

from ger_wiki.predictor import TextPredictor
from ger_wiki.reader import GerReader

st.header("Geographic Entity Recognition")

model_name = st.selectbox(
    "Choose model.",
    (
        "wiki_bert_model",
        "wiki_crf_basic_model",
        "wiki_crf_model",
        "wiki_distil_model",
        "wiki_roberta_model",
    ),
)
ARCHIVE_PATH = f"models/{model_name}/model.tar.gz"


@st.cache(allow_output_mutation=True)
def load_predictor(archive_path):
    archive = load_archive(archive_path)
    predictor = TextPredictor.from_archive(archive, predictor_name="text_predictor")
    predictor.cuda_device = 0 if torch.cuda.is_available() else -1
    return predictor


def run_model(predictor, passage):
    result = predictor.predict(passage)
    spans = bioul_tags_to_spans(result["tags"])

    passage_tokens = result["words"]

    tags = []
    start = []
    end = []
    for r in spans:
        tags.append(r[0])
        start.append(r[1][0])
        end.append(r[1][1])

    places = []
    for span in spans:
        start, end = span[1]

        if span[0] == "PLACE_NAM":
            place = " ".join(passage_tokens[start : end + 1])
            places.append(place)

        passage_tokens[start : end + 1] = [
            "**" + token + "**" for token in passage_tokens[start : end + 1]
        ]
        # passage_tokens[end] += " _<sub>" + span[0].split("_")[1] + "</sub>_"
        passage_tokens[end] += " _<sub>PLACE</sub>_"
    return passage_tokens, places


passage = st.text_area(
    "sentence",
    "Headingley is a suburb of Leeds, West Yorkshire, England, approximately two miles out of the city centre, to the north west along the A660 road. Headingley is the location of the Beckett Park campus of Leeds Beckett University and Headingley Stadium.",
)

click_model = st.button("Run Model")
if click_model:
    predictor = load_predictor(ARCHIVE_PATH)
    passage_tokens, places = run_model(predictor, passage)

    passage_tokens = " ".join(passage_tokens)
    # remove space from brackets
    passage_tokens = re.sub(r"\s([\)])", r"\1", passage_tokens)
    passage_tokens = re.sub(r"([\(])\s", r"\1", passage_tokens)
    # and punctuation
    passage_tokens = re.sub(r'\s([?.!",\'])', r"\1", passage_tokens)
    st.markdown(passage_tokens, unsafe_allow_html=True)
