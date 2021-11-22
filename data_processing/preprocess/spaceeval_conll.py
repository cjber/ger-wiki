import os
import random
from typing import List
from xml.etree import ElementTree

import spacy
import tqdm
from spacy.training import offsets_to_biluo_tags


class SpaceConll:
    def __init__(self, language: str = "en"):
        """
        Converts ISO-Space formatted .xml annotations to the CoNLL format.

        :param language str: Spacy language.
        """
        self.language = language

    def output_conll(self, input_dir: str, output_dir: str, split=0.1):
        """
        Take multiple input files and write to single CoNLL file.

        :param input_dir str: Input ISO-Space directory.
        :param output_dir str: Output CoNLL formatted directory.
        :param split int: Train/test split size.
        """
        nlp = spacy.blank(self.language)
        file_list = set()
        for _, _, file in list(os.walk(input_dir)):
            for data_file in tqdm.tqdm(file):
                if data_file.endswith("xml"):
                    file_list.add(data_file)
        test_files = set(random.sample(file_list, int(split * len(file_list) // 1)))
        train_files = file_list - test_files

        with open(output_dir / "spaceeval_test.conll", "w") as fp:
            for file in test_files:
                file_path = os.path.join(input_dir, file)
                conll_formatted = self.spaceeval_to_conll(file_path, nlp)
                fp.write("\n".join(f"{x[0]} {x[1]}" for x in conll_formatted))

        with open(output_dir / "spaceeval_train.conll", "w") as fp:
            for file in train_files:
                file_path = os.path.join(input_dir, file)
                conll_formatted = self.spaceeval_to_conll(file_path, nlp)
                fp.write("\n".join(f"{x[0]} {x[1]}" for x in conll_formatted))

    def spaceeval_to_conll(self, spaceeval_xml_file: str, nlp: str):
        """
        Convert ISO-Space formatted file to CoNLL format.

        :param spaceeval_xml_file str: ISO-Space formatted XML file.
        :param nlp spacy.long.en.English: English SpaCy language model.
        """
        root = ElementTree.parse(spaceeval_xml_file).getroot()

        text: str = root.find("TEXT").text
        tags: List = list(root.find("TAGS"))

        offset = 0
        sent_tokens = []
        sent_ents = []
        for sent in text.split(". "):
            sent = sent + ". "

            # split sentences by newlines
            sent_nlp = nlp(sent)
            tokens = [str(token) for token in sent_nlp]
            spatial_entities = self.extract_labels(tags, sent, offset)

            ent_biluo = offsets_to_biluo_tags(sent_nlp, spatial_entities)

            # allennlp cant handle unknown tags so just use other
            ent_biluo = ["O" if x == "-" else x for x in ent_biluo]

            sent_tokens.extend(tokens)
            sent_tokens.append("")
            sent_ents.extend(ent_biluo)
            sent_ents.append("\n")
            offset += len(sent)

        file_conll = list(zip(sent_tokens, sent_ents))

        for pair in file_conll:
            if "\n" in pair[0] or "\u2002" in pair[0] or " " in pair[0]:
                file_conll.remove(pair)
        return file_conll

    def extract_labels(self, tags: List, sent: str, offset: int):
        """
        Create entity offsets from list of tags from xml file.

        :param tags List: List of xml tags.
        :param sent str: Sentence containing entities.
        :param offset int: Sentence offset in document.
        """
        ent_labels = ["PATH", "PLACE"]
        ents = [
            tag
            for tag in tags
            if tag.tag in ent_labels
            if int(tag.attrib["end"]) <= len(sent) + offset
            if int(tag.attrib["start"]) >= offset
        ]
        ents = [
            (
                int(ent.attrib["start"]) - (offset),
                int(ent.attrib["end"]) - (offset),
                # change paths to place
                "PLACE" + "_" + str(ent.attrib["form"]),
            )
            for ent in ents
            # very low accuracy for places without NAM/NOM
            if str(ent.attrib["form"]) != ""
        ]
        return ents
