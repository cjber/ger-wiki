library(tidyverse)
library(kableExtra)
library(here)

oog <- read_csv(here("data_processing/data/results/distil_oog.csv"))
oog <- oog[!oog$index %in% c("United Kingdom", "UK", "Scottish"), ]

oog_s <- read_csv(here("data_processing/data/results/spacy_oog.csv")) |>
    arrange(-count) |>
    tibble::rowid_to_column("IDX")
oog_s <- oog_s[!oog_s$place %in% c("United Kingdom", "UK", "Scottish"), ]


nam_t <- oog %>%
    arrange(desc(place)) %>%
    filter(place > 100) %>%
    {
        rbind(head(.), "...", tail(.))
    } %>%
    rename("IDX" = "...1", "Place (DistilBERT)" = "index", "Count" = "place")

mor_t <- oog_s %>%
    arrange(desc(count)) %>%
    filter(count > 100) %>%
    {
        rbind(head(.), "...", tail(.))
    } %>%
    rename("Place (SpaCy)" = "place", "Count" = "count")

list(nam_t, mor_t) %>%
    knitr::kable(
        format = "latex",
        booktabs = TRUE,
        linesep = "",
        caption = "Top and bottom named places by frequency, excluding any present in the GeoNames gazetter or mentioned less than 100 times."
    ) %>%
    kableExtra::kable_styling(font_size = 9)
