<div align="center">

# Transformer based named entity recognition for place name extraction

<a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/></a>
<a href="https://allennlp.org"><img alt="AllenNLP" src="https://img.shields.io/badge/AllenNLP%20-0000004C.svg?&style=for-the-badge&color=blue"/></a>  
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/style-black-000000.svg?style=flat-square"></a>

**DOI:** [10.1080/13658816.2022.2133125](https://www.tandfonline.com/doi/full/10.1080/13658816.2022.2133125)

</div>


[Cillian
Berragan](https://www.liverpool.ac.uk/geographic-data-science/our-people/)
\[[`@cjberragan`](http://twitter.com/cjberragan)\]<sup>1\*</sup> [Alex
Singleton](https://www.liverpool.ac.uk/geographic-data-science/our-people/)
\[[`@alexsingleton`](https://twitter.com/alexsingleton)\]<sup>1</sup>
[Alessia
Calafiore](https://www.liverpool.ac.uk/geographic-data-science/our-people/)
\[[`@alel_domi`](http://twitter.com/alel_domi)\]<sup>1</sup> [Jeremy
Morley](https://ordnancesurvey.co.uk/)
\[[`@jeremy_morley`](https://twitter.com/jeremy_morley)]<sup>2</sup>

<sup>1</sup> _Geographic Data Science Lab, University of Liverpool,
Liverpool, United Kingdom_  
<sup>2</sup> _Ordnance Survey Limited, Explorer House, Adanac Drive,
Nursling, Southampton, United Kingdom_

<sup>\*</sup> _Correspondence_: C.Berragan@liverpool.ac.uk

## Abstract

Place names embedded in online natural language text present a useful source of geographic information. Despite this, many methods for the extraction of place names from text use pre-trained models that were not explicitly designed for this task. Our paper builds five custom-built Named Entity Recognition (NER) models, and evaluates them against three popular pre-built models for place name extraction. The models are evaluated using a set of manually annotated Wikipedia articles with reference to the F~1~ score metric. Our best performing model achieves an F~1~ score of 0.939 compared with 0.730 for the best performing pre-built model. Our model is then used to extract all place names from Wikipedia articles in Great Britain, demonstrating the ability to more accurately capture unknown place names from volunteered sources of online geographic information.

## Description

A fine-tuned DistilBERT transformer model is presented for the identification of place names from text. This repository contains the code used to build the model using [AllenNLP](https://allennlp.org/) and the code used to compare the fine-tuned model against existing models used in geoparsing systems.

## Project layout

```bash
ger_wiki
├── configs  # allennlp model configurations
│   ├── wiki_bert.jsonnet
│   ├── wiki_crf_basic.jsonnet
│   ├── wiki_crf.jsonnet
│   ├── wiki_distil.jsonnet
│   └── wiki_roberta.jsonnet
├── data_processing  # scripts relating to data processing
│   ├── preprocess
│   │   ├── dbpedia_query.py
│   │   └── doccano_functions.py
│   ├── oog
│   │   └── distil_oog.py
│   │   └── spacy_oog.py
├── Dockerfile
├── eval  # scripts relating to evaluation between models
│   ├── crossval.py
│   ├── significance.py
│   └── visualise.py
├── ger_streamlit.py  # model dashboard
├── ger_wiki  # allennlp model scripts
│   ├── batch_predictor.py
│   ├── optimisation.py
│   ├── predictor.py
├── main.py  # main cli script
├── paper
└── tests
```

## Recreate software environment

1. **Using a Docker image**

   - `docker build . -t ger_wiki`
   - `docker run -it --rm --gpus all ger_wiki` If the machine has no GPU, run `docker run -it --rm ger_wiki`

2. **Using a Python virtual environment**

   - Extract `ger_wiki.tar.gz` (or use `git clone`)
   - Ensure the current python version is 3.8.5, `pyenv` is a useful tool for this.
   - Virtual environment configuration may either use the `requirements.txt` file (e.g. with `conda/venv`), or with the `pyproject.toml` file with `poetry`.
   - Run `Rscript paper/setup.R` from the base directory. This will install the R dependencies used for figure creation. R major version 4 is likely required. Several external libraries are required for some R packages (e.g. gdal). If required, the `Dockerfile` may be consulted to see what may be needed.

Assuming a correct environment setup, both methods should yield the same results in the following sections.

## Train entity recognition models

All CoNLL formatted data used to train the entity recognition models is included in `data_processing/data/processed`, allowing the models to run. These models have much of their configuration contained within the `configs/` directory as `.jsonnet` files. These contain hyperparameters and model configurations with one file per model.

> **WARNING: These commands will likely require a GPU**

- Run `python3 main.py wiki_distil`: Train + evaluate the DistilBERT model on Wikipedia data. Replace `wiki_distil` with any other configuration in the `configs/` dir for other models (e.g. `configs/wiki_{model}.jsonnet`)

Once models have finished running, the metrics shown on Table 3 are given as output. This output is also given within the `models/{model}` directories as `metrics.json`. As Table 3 utilises these `.json` files for creation, the metrics in the main paper will always be the most up to date.

### Main source code

The main setup for these models is contained within the `ger_wiki` directory as python scripts.

- `reader.py` reads in the `.conll` files as a format readable for these models (`Instances`), it is also designed to preserve metadata associated with the text (e.g. Wikipedia article title).

- `predictor.py` contains a single instance predictor that is able to read plain text into a trained model to output results. This class is used by the `ger_streamlit.py` file which may be used to demo the model.

- `batch_predictor.py` contains a class used by `main.py` to read in the Wikipedia `CSV` that was queried from DBpedia. This class parallelises batches of text and outputs the place names identified by the model into a further `CSV` file.

## Reproduce dataset of Wikipedia place names

The main place names dataset is provided in `data_processing/data/results/predictions.csv` and may be recreated in this section. If the `wiki_distil` model is trained fully as above, this following section will work. However, if model training was not possible, the provided `model.tar.gz` will need to be added to `models/wiki_distil/`. For the Docker container, run this command from another shell while the container is running:

> **INFO:** To find `container_id` run `docker ps`

- `docker cp model.tar.gz (container_id):/project/models/wiki_distil/`

To create predictions (Runs either on CPU or GPU):

#### Full dataset

1. Run `python3 main.py wiki_distil --predict`: use the trained Wiki model to extract place names from all Wikipedia articles queried from Wikipedia. Will be **very** slow to complete on CPU. This relies on `data_processing/data/raw/wiki/wiki_info.csv` that was created by the DBpedia query.

#### Interactive demo

2. Run `streamlit run ger_streamlit.py`: host an interactive app to explore outputs from the model. This may be used to recreate images shown in Section 5.

## Recreate figures

All code relating to table and figure production is contained within `paper/tables` and `paper/figures` respectively. Changes to results will also update all figures and tables. Each `R` script may be run independently as they are completely self-contained.

- For example run: `Rscript paper/figures/figure2_wiki_dist.R`. This may not produce output, it is likely easier to reproduce these figures using an R REPL.

The paper itself uses R Markdown to render figures and tables directly with text, source code, and data, to knit this document and produce an updated PDF:

- `Rscript -e 'rmarkdown::render("paper/main.Rmd")'`

> **NOTE:** If models have partially completed, files required for the creation of figures and tables may have been removed. If this occurs it is likely best to restart the Docker container, or start with the fresh archive.

## Pre-processing

Contained within `data_processing/preprocess` are three python scripts. To evaluate the preprocessing it is best to inspect these files manually as they were not intended to be run sequentially.

To ensure functions were correctly implemented `pytest` may be run from the base directory which uses toy data for the functions used here. Running `pytest` from the base directory should not throw errors. The `tests` directory contains the toy data, and unit tests relating to this preprocessing.

The three preprocessing scripts are:

- `data_processing/preprocess/dbpedia_query.py`: DBpedia query and text cleaning.

- `data_processing/preprocess/doccano_functions.py`: Helper functions to convert model predictions to Doccano formatted files and these to CoNLL.

These files were not run more than once and are only kept for completeness to demonstrate the data preprocessing involved. The query script may be ran, to retrieve a more up to date Wikipedia corpus, but will be very slow.

- `python -m data_processing.preprocess.dbpedia_query` will update the raw Wikipedia data.

## Evaluation

The directory `eval` contains two scripts for the evaluation of the chosen model against pre-built solutions. Please note that the pre-built models must be downloaded first. `python -m spacy download en_core_web_sm`, `python -m spacy download en_core_web_lg`, Stanza’s model may be installed using python `import stanza; stanza.download('en')`.

- `crossval.py`: Obtain F<sub>1</sub> scores for 10 subsets of the validation data for each model.

- `significance.py`: Significance testing between models.

- `visualise.py`: Visualises the NER output from pre-built models + DistilBERT.

## Out of gazetteer

`data_processing/oog` contains the processing which identifies place names that do not appear in the GeoNames gazetteer for the DistilBERT model and Spacy. Used in Table 5.

## Dockerfile

The included Dockerfile may be used to build the Docker image from
scratch:

- `docker build -t ger_wiki .`

- `docker run -it --rm --gpus all ger_wiki`
