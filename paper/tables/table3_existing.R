library(tidyverse)
library(lubridate)
library(jsonlite)
library(here)
library(kableExtra)

existing_ner <- read_csv(
    here("data_processing/data/results/crossval.csv")
) %>%
    select(-...1) %>%
    group_by(name) %>%
    summarise_if(is.numeric, funs(mean, sd)) %>%
    arrange(desc(f1_mean)) %>%
    mutate_at(
        c(
            "accuracy_mean",
            "precision_mean",
            "recall_mean",
            "f1_mean"
        ),
        format,
        digits = 3, nsmall = 3
    ) %>%
    mutate_at(c(
        "accuracy_sd",
        "precision_sd",
        "recall_sd",
        "f1_sd"
    ), format, digits = 2) %>%
    mutate(
        accuracy = paste0(accuracy_mean, " ±", accuracy_sd),
        precision = paste0(precision_mean, " ±", precision_sd),
        recall = paste0(recall_mean, " ±", recall_sd),
        f1 = paste0(f1_mean, " ±", f1_sd),
    ) %>%
    select(-c(ends_with("mean"), ends_with("sd"))) %>%
    as.data.frame()

rownames(existing_ner) <- existing_ner[, 1]
existing_ner <- existing_ner %>% select(-c(name))

metric_names <- c(
    "Accuracy",
    "Precision",
    "Recall",
    "F1"
)
colnames(existing_ner) <- metric_names

metrics <- existing_ner %>%
    mutate_if(is.numeric, format, digits = 3, nsmall = 3) %>%
    as.data.frame() %>%
    arrange(desc(F1))

metrics$F1[1:3] <- paste0("\\textbf{", as.character(metrics$F1[1:3]), "}")

metrics %>%
    cjrmd::make_latex_table(
        caption = "Geographic entity recognition mean (±SD) performance metrics over 3 runs of annotated Wikipedia test data subsets. Pre-built NER models are shown in italics. Bold values indicate statistically significant F1 scores of fine-tuned models in relation to `Stanza` (Paired t-tests $p<0.05$)."
    ) %>%
    row_spec(5, hline_after = TRUE) %>%
    row_spec(6:8, italic = TRUE)
