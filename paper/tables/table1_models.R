library(tidyverse)
library(kableExtra)

info <- c("Name", "Embeddings", "Intermediate", "Output", "Optimiser")
blstm <- c("\\texttt{BiLSTM-CRF (Basic)}", "Token \\{50\\}", "2-layer BiLSTM \\{200\\}", "CRF", "Adam")
lstm <- c("\\texttt{BiLSTM-CRF}", linebreak("GloVe Token \\{50\\}\n Character \\{16\\}"), "2-layer BiLSTM \\{200\\}", "CRF", "Adam")
bert <- c("\\texttt{BERT}", "BERT \\{768\\}", "12-layer Transformer \\{768\\}", "CRF", "AdamW")
roberta <- c("\\texttt{RoBERTa}", "RoBERTa \\{768\\}", "12-layer Transformer \\{768\\}", "CRF", "AdamW")
distilbert <- c("\\texttt{DistilBERT}", "DistilBERT \\{768\\}", "6-layer Transformer \\{768\\}", "CRF", "AdamW")

tab <- cbind(blstm, lstm, bert, roberta, distilbert) %>%
    t() %>%
    as.data.frame()
names(tab) <- info
row.names(tab) <- NULL

tab %>%
    cjrmd::make_latex_table(caption = "Overview of the models trained through our paper, detailing the architecture used. Integers in \\{ \\} indicate the vector dimensions", align = c("l", "l", "l", "l", "l"))
