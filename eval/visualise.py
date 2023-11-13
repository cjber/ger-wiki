import spacy
import spacy_stanza
import stanza
from stanza.pipeline.core import ResourcesFileNotFoundError

text_china = u"Originally having participated in Olympics as the delegation of the Republic of China (ROC) from 1924 (Summer Olympics) to 1976 (Winter Olympics), China competed at the Olympic Games under the name of the People's Republic of China (PRC) for the first time in 1952, at the Summer Games in Helsinki, Finland."

text_rowlatts = u"The suburb is roughly bordered by Spencefield Lane to the east and Whitehall Road to the south, which separates it from neighbouring Evington. A second boundary within the estate consists of Coleman Road to Ambassador Road through to Green Lane Road; Rowlatts Hill borders Crown Hills to the west. To the north, at the bottom of Rowlatts Hill is Humberstone Park which is located within Green Lane Road, Ambassador Road and also leads on to Uppingham Road (the A47), which is also Rowlatts Hill."

try:
    nlp_stanza = spacy_stanza.load_pipeline(name="en", processors="tokenize,ner")
except ResourcesFileNotFoundError:
    stanza.download("en")
    nlp_stanza = spacy_stanza.load_pipeline(name="en", processors="tokenize,ner")
doc_stanza = nlp_stanza(text_rowlatts)

nlp_spacy = spacy.load("en_core_web_lg")
doc_spacy = nlp_spacy(text_rowlatts)
# spacy.displacy.serve(doc_spacy, style="ent")
spacy.displacy.serve(doc_stanza, style="ent")
