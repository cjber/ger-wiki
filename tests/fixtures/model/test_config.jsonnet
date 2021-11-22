{
    "dataset_reader": {
        "type": "ger_reader",
        "token_indexers": {
            "tokens": {
                "type": "single_id"
            }
        },
    },
    "train_data_path": "tests/fixtures/reader/toy_data.txt",
    "validation_data_path": "tests/fixtures/reader/toy_data.txt",
    "data_loader": {
        "batch_size": 8,
        "shuffle": true
    },
    "model": {
        "type": "crf_tagger",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 10
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": 10,
            "hidden_size": 1
        }
    },
    "trainer": {
        "cuda_device": -1,
        "optimizer": "adam",
        "num_epochs": 5
    }
}
