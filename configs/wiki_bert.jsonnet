local base_dir = "data_processing/data";

local transformer_model = "bert-base-cased";
local transformer_hidden_dim = 768;
local max_length = 128;
local epochs = 50;
local seed = 42;

local lr = 2e-5;
local batch_size = 8;
local weight_decay = 0.01;
local dropout = 0.1;

local patience = 8;
local lr_patience = 2;
local use_amp = true;
local eps = 1e-8;
local grad_norm = 1.0;

{
    "numpy_seed": seed,
    "pytorch_seed": seed,
    "random_seed": seed,
    "dataset_reader": {
        "type": "ger_reader",
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer_mismatched",
                "model_name": transformer_model,
                "max_length": max_length
            },
        },
    },
    "train_data_path": base_dir + "/processed/wiki_train.conll",
    "validation_data_path": base_dir + "/processed/wiki_val.conll",
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": batch_size
        }
    },
    "model": {
        "type": "crf_tagger",
        "calculate_span_f1": true,
        "constrain_crf_decoding": true,
        "dropout": dropout,
        "include_start_end_transitions": false,
        "label_encoding": "BIOUL",
        "verbose_metrics": true,
        "encoder": {
            "type": "pass_through",
            "input_dim": transformer_hidden_dim,
        },
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer_mismatched",
                    "model_name": transformer_model,
                    "max_length": max_length
                }
            }
        },
    },
    "trainer": {
        "cuda_device": 0,
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": lr,
            "weight_decay": weight_decay,
            "parameter_groups": [
                [
                    [
                        "bias",
                        "LayerNorm\\.weight",
                        "layer_norm\\.weight"
                    ],
                    {
                        "weight_decay": 0
                    }
                ]
            ],
            "correct_bias": true,
            "eps": eps
        },
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "patience": lr_patience,
            "mode": "max"
        },
        "num_epochs": epochs,
        "validation_metric": "+f1-measure-overall",
        "patience": patience,
        "grad_norm": grad_norm,
        "use_amp": use_amp
    }
}
