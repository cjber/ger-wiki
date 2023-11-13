local base_dir = "data_processing/data";
local seed = 42;
local batch_size = 8;
local epochs = 50;
local patience = 8;
local use_amp = false;
local grad_norm = 1.0;

{
    "numpy_seed": seed,
    "pytorch_seed": seed,
    "random_seed": seed,
    "dataset_reader": {
        "type": "ger_reader",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 3
      }
    }
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
    "label_encoding": "BIOUL",
    "constrain_crf_decoding": true,
    "calculate_span_f1": true,
    "dropout": 0.5,
    "include_start_end_transitions": false,
    "verbose_metrics": true,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 50,
            "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.50d.txt.gz",
            "trainable": true
        },
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
                "embedding_dim": 16
            },
            "encoder": {
                "type": "cnn",
                "embedding_dim": 16,
                "num_filters": 128,
                "ngram_filter_sizes": [3],
                "conv_layer_activation": "relu"
            }
          }
       },
    },
    "encoder": {
        "type": "lstm",
        "input_size": 50 + 128,
        "hidden_size": 200,
        "num_layers": 2,
        "dropout": 0.5,
        "bidirectional": true
    },
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": 0.001
    },
    "checkpointer": {
        "num_serialized_models_to_keep": 1,
    },
    "validation_metric": "+f1-measure-overall",
    "num_epochs": epochs,
    "grad_norm": grad_norm,
    "patience": patience,
    "use_amp": use_amp,
  }
}
