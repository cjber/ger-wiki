library(tidyverse)
library(kableExtra)

info <- c("Name", "Training Data", "Architecture", "Reported NER $F_{1}$")
spacy_sm <- c("SpaCy (small)", "OntoNotes 5", "CNN", paste0(0.84, "$^a$"))
spacy_lg <- c("SpaCy (large)", "OntoNotes 5", "CNN", paste0(0.85, "$^a$"))
stanza <- c("Stanza", "OntoNotes 5", "BiLSTM CRF", paste0(0.89, "$^b$"))

tab <- cbind(spacy_sm, spacy_lg, stanza) %>%
    t() %>%
    as.data.frame()
names(tab) <- info
row.names(tab) <- NULL

tab %>%
    cjrmd::make_latex_table(caption = "Pre-built NER models", align = c("l", "l", "l", "c", "c")) %>%
    footnote(general_title = "", general = c("a https://spacy.io/models/en", "b https://stanfordnlp.github.io/stanza/performance.html"))
