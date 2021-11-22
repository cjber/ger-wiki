import spacy
import spacy_stanza
import stanza
from stanza.pipeline.core import ResourcesFileNotFoundError

text = u"Originally having participated in Olympics as the delegation of the Republic of China (ROC) from 1924 (Summer Olympics) to 1976 (Winter Olympics), China competed at the Olympic Games under the name of the People's Republic of China (PRC) for the first time in 1952, at the Summer Games in Helsinki, Finland."

try:
    nlp_stanza = spacy_stanza.load_pipeline(name="en", processors="tokenize,ner")
except ResourcesFileNotFoundError:
    stanza.download("en")
    nlp_stanza = spacy_stanza.load_pipeline(name="en", processors="tokenize,ner")
doc_stanza = nlp_stanza(text)

nlp_spacy = spacy.load("en_core_web_lg")
doc_spacy = nlp_spacy(text)
spacy.displacy.serve(doc_spacy, style="ent")
spacy.displacy.serve(doc_stanza, style="ent")
