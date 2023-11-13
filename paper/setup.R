if (!require("pacman")) install.packages("pacman", repos = "http://cran.us.r-project.org")
library(pacman)

pkgs <- c(
    "here",
    "tidyverse",
    "rmarkdown",
    "tinytex",
    "rticles",
    "kableExtra",
    "showtext",
    "bibtex",
    "reticulate",
    "patchwork",
    "jsonlite",
    "sf",
    "sfheaders",
    "viridis",
    "ggthemes",
    "ggrastr",
    "lubridate",
    "R.utils",
    "magick",
    "cowplot"
)

pacman::p_load(pkgs, character.only = TRUE)

if (!require("cjrmd")) remotes::install_github("cjber/cjrmd")
ggplot2::theme_set(cjrmd::cj_plot_theme)
