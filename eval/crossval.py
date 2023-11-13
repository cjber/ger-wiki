import itertools
import random

import pandas as pd
import spacy
import stanza
import torch
from allennlp.data.dataset_readers.dataset_utils import bioul_tags_to_spans
from allennlp.models.archival import load_archive
from seqeval.metrics import accuracy_score, classification_report
from seqeval.scheme import BILOU
from spacy.tokens import Doc
from spacy.training.iob_utils import iob_to_biluo

from ger_wiki.predictor import TextPredictor

spacy.util.fix_random_seed(52)


def load_predictor(archive_path):
    archive = load_archive(archive_path)
    predictor = TextPredictor.from_archive(archive, predictor_name="text_predictor")
    predictor.cuda_device = 0 if torch.cuda.is_available() else -1
    return predictor


DISTIL = "models/wiki_distil_model/model.tar.gz"
BERT = "models/wiki_bert_model/model.tar.gz"
ROBERTA = "models/wiki_roberta_model/model.tar.gz"
CRF_LSTM = "models/wiki_crf_model/model.tar.gz"
CRF_BASIC = "models/wiki_crf_basic_model/model.tar.gz"

nlp_spacy_sm = spacy.load("en_core_web_sm")
nlp_spacy_lg = spacy.load("en_core_web_lg")
nlp_spacy_sm.disable_pipes("tagger", "parser", "lemmatizer")
nlp_spacy_lg.disable_pipes("tagger", "parser", "lemmatizer")
nlp_stanza = stanza.Pipeline(
    lang="en", processors="tokenize,ner", tokenize_pretokenized=True
)
nlp_distil = load_predictor(DISTIL)
nlp_bert = load_predictor(BERT)
nlp_roberta = load_predictor(ROBERTA)
nlp_crf = load_predictor(CRF_LSTM)
nlp_basic = load_predictor(CRF_BASIC)

file_path = "data_processing/data/processed/wiki_test.conll"

conll = []
with open(file_path, "r") as conll_file:
    for divider, lines in itertools.groupby(
        conll_file, lambda line: line.strip() == ""
    ):
        if divider:
            continue
        fields = [line.strip().split() for line in lines]
        fields = [line for line in zip(*fields)]
        tokens, ner_tags = fields
        conll.append((tokens, ner_tags))


def bioes_to_biluo(tags):
    biluo_tags = []
    for tag in tags:
        if tag[0] == "S":
            biluo_tags.append("B" + tag[1:])
        elif tag[0] == "E":
            biluo_tags.append("L" + tag[1:])
        else:
            biluo_tags.append(tag)
    return biluo_tags


def stanza_process(tokens):
    stanza_doc = nlp_stanza([[*tokens]]).to_dict()[0]
    stanza_preds = [token["ner"] for token in stanza_doc]
    stanza_preds = bioes_to_biluo(stanza_preds)
    stanza_preds = iob_to_biluo(stanza_preds)
    stanza_preds = [
        tag[:2] + "PLACE" if tag[2:] in ["GPE", "LOC", "FAC"] else "O"
        for tag in stanza_preds
    ]
    return stanza_preds


def spacy_process(tokens, nlp_spacy):
    text = " ".join(tokens)
    tokens_dict = {text: tokens}

    nlp_spacy.tokenizer = lambda _: Doc(nlp_spacy.vocab, tokens_dict[text])
    spacy_doc = nlp_spacy(text)

    spacy_preds = []
    for token in spacy_doc:
        if token.ent_type_ in ["GPE", "LOC", "FAC"]:
            spacy_preds.append(token.ent_iob_ + "-" + "PLACE")
        else:
            spacy_preds.append("O")
    return iob_to_biluo(spacy_preds)


def create_report(gold_tags, pred_tags):
    report = classification_report(
        y_true=gold_tags,
        y_pred=pred_tags,
        mode="strict",
        scheme=BILOU,
        output_dict=True,
        zero_division=0,
    )

    overall_score = report.pop("micro avg")

    return {
        "accuracy": accuracy_score(y_true=gold_tags, y_pred=pred_tags),
        "precision": overall_score["precision"],
        "recall": overall_score["recall"],
        "f1": overall_score["f1-score"],
    }


conll_subsets = [conll[i::3] for i in range(3)]

spacy_report_sm = []
spacy_report_lg = []
stanza_report = []
distil_report = []
bert_report = []
roberta_report = []
crf_report = []
crf_basic_report = []
for subset in conll_subsets:
    spacy_tags_sm = []
    spacy_tags_lg = []
    stanza_tags = []
    distil_tags = []
    bert_tags = []
    roberta_tags = []
    crf_tags = []
    crf_basic_tags = []

    gold_tags = []
    for example in subset:
        tags = [tag[:-4] if tag[2:] == "PLACE_NAM" else "O" for tag in example[1]]

        stanza_preds = stanza_process(example[0])
        spacy_preds_sm = spacy_process(example[0], nlp_spacy_sm)
        spacy_preds_lg = spacy_process(example[0], nlp_spacy_lg)
        distil_preds = nlp_distil.predict(" ".join(example[0]))
        bert_preds = nlp_bert.predict(" ".join(example[0]))
        roberta_preds = nlp_roberta.predict(" ".join(example[0]))
        crf_preds = nlp_crf.predict(" ".join(example[0]))
        crf_basic_preds = nlp_basic.predict(" ".join(example[0]))

        assert (
            len(tags)
            == len(spacy_preds_sm)
            == len(spacy_preds_lg)
            == len(stanza_preds)
            == len(distil_preds["tags"])
        )
        distil_preds["tags"] = [
            tag[:-4] if tag[2:] == "PLACE_NAM" else "O" for tag in distil_preds["tags"]
        ]
        bert_preds["tags"] = [
            tag[:-4] if tag[2:] == "PLACE_NAM" else "O" for tag in bert_preds["tags"]
        ]
        roberta_preds["tags"] = [
            tag[:-4] if tag[2:] == "PLACE_NAM" else "O" for tag in roberta_preds["tags"]
        ]
        crf_preds["tags"] = [
            tag[:-4] if tag[2:] == "PLACE_NAM" else "O" for tag in crf_preds["tags"]
        ]
        crf_basic_preds["tags"] = [
            tag[:-4] if tag[2:] == "PLACE_NAM" else "O"
            for tag in crf_basic_preds["tags"]
        ]
        gold_tags.append(tags)
        spacy_tags_sm.append(spacy_preds_sm)
        spacy_tags_lg.append(spacy_preds_lg)
        stanza_tags.append(stanza_preds)
        distil_tags.append(distil_preds["tags"])
        bert_tags.append(bert_preds["tags"])
        roberta_tags.append(roberta_preds["tags"])
        crf_tags.append(crf_preds["tags"])
        crf_basic_tags.append(crf_basic_preds["tags"])

    spacy_report_sm.append(create_report(gold_tags, spacy_tags_sm))
    spacy_report_lg.append(create_report(gold_tags, spacy_tags_lg))
    stanza_report.append(create_report(gold_tags, stanza_tags))
    distil_report.append(create_report(gold_tags, distil_tags))
    bert_report.append(create_report(gold_tags, bert_tags))
    roberta_report.append(create_report(gold_tags, roberta_tags))
    crf_report.append(create_report(gold_tags, crf_tags))
    crf_basic_report.append(create_report(gold_tags, crf_basic_tags))


stanza_df = pd.DataFrame(stanza_report)
stanza_df["name"] = "Stanza"
spacy_sm_df = pd.DataFrame(spacy_report_sm)
spacy_sm_df["name"] = "SpaCy (small)"
spacy_lg_df = pd.DataFrame(spacy_report_lg)
spacy_lg_df["name"] = "SpaCy (Large)"
distil_df = pd.DataFrame(distil_report)
distil_df["name"] = "DistilBERT"
bert_df = pd.DataFrame(bert_report)
bert_df["name"] = "BERT"
roberta_df = pd.DataFrame(roberta_report)
roberta_df["name"] = "RoBERTa"
crf_df = pd.DataFrame(crf_report)
crf_df["name"] = "CRF biLSTM"
crf_basic_df = pd.DataFrame(crf_basic_report)
crf_basic_df["name"] = "CRF biLSTM (basic)"

pd.concat(
    [
        stanza_df,
        spacy_sm_df,
        spacy_lg_df,
        distil_df,
        bert_df,
        roberta_df,
        crf_df,
        crf_basic_df,
    ]
).to_csv("data_processing/data/results/crossval.csv")
