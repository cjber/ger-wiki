library(tidyverse)
library(lubridate)
library(jsonlite)
library(here)
library(kableExtra)

format_metrics <- function(model) {
    metrics <- read_json(
        paste0(here(paste0("models/", model, "/metrics.json")))
    ) %>%
        as.data.frame() %>%
        select(
            training_duration,
            best_validation_f1.measure.overall
        ) %>%
        mutate_if(is.numeric, format, digits = 3, nsmall = 3) %>%
        mutate(
            training_duration = as.integer(as.numeric(hms(training_duration)))
        ) %>%
        tibble()

    model_size <- paste0("models/", model, "/best.th") %>%
        here() %>%
        file.size() %>%
        as.numeric() %>%
        `/`(1000000) %>%
        round(1)

    metrics["Model Size"] <- format(model_size, nsmall = 1)
    metrics <- metrics %>% select(c(3, 1, 2))
    return(metrics)
}

crf <- format_metrics("wiki_crf_model")
crf_basic <- format_metrics("wiki_crf_basic_model")
bert <- format_metrics("wiki_bert_model")
distil <- format_metrics("wiki_distil_model")
roberta <- format_metrics("wiki_roberta_model")

metric_table <- rbind(crf, crf_basic, bert, distil, roberta) %>%
    as.data.frame()
row.names(metric_table) <- c(
    "CRF", "CRF (basic)", "BERT", "DistilBERT", "RoBERTa"
)
names(metric_table) <- c(
    "Size (MB)", "Time (S)",
    "F1 Overall"
)

metric_table <- metric_table %>% arrange(desc(`F1 Overall`))

metric_table[
    which.max(metric_table[, "F1 Overall"]), "F1 Overall"
] <- cell_spec(metric_table[
    which.max(metric_table[, "F1 Overall"]), "F1 Overall"
], bold = TRUE)

metric_table$`F1 Overall`[4:5] <- paste0(as.character(metric_table$`F1 Overall`[4:5]), "†")

metric_table %>%
    cjrmd::make_latex_table(
        caption = "Model test data performance metrics for each model trained on annotated Wikipedia data. Best scores are in bold. † indicates a significant difference in F1 score with respect to `DistilBERT`.",
        align = c("l", "l", "c", "c")
    ) %>%
    column_spec(4, border_left = TRUE)
