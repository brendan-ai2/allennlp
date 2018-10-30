{
  "dataset_reader": {
    "type": "multiprocess",
    "base_reader": {
        "type": "simple_language_modeling",
        "token_indexers": {
          "tokens": {
            "type": "single_id",
            "start_tokens": ["<s>"],
            "end_tokens": ["</s>"]
          },
          "token_characters": {
            "type": "characters",
            "start_tokens": ["<s>"],
            "end_tokens": ["</s>"]
          }
        }
    },
    "num_workers": 8
    # TODO(brendanr): Consider epochs_per_read and output_queue_size.
  },
  "train_data_path": "/home/brendanr/workbenches/calypso/train/*",
  "validation_data_path": "/home/brendanr/workbenches/calypso/dev/*",
  "vocabulary": {
      "tokens_to_add": {
          "tokens": ["<s>", "</s>"],
          "token_characters": ["<>/s"]
      },
  },
  "model": {
    "type": "bidirectional-language-model",
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "token_embedders": {
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
                "num_embeddings": 262,
                "embedding_dim": 4
            },
            "encoder": {
                "type": "cnn-highway",
                "activation": "relu",
                "embedding_dim": 4,
                "filters": [[1, 4], [2, 8], [3, 16], [4, 32], [5, 64]],
                "num_highway": 2,
                "projection_dim": 16,
                "projection_location": "after_cnn"
            }
        }
      }
    },
    "contextualizer": {
        "type": "lstm",
        "bidirectional": true,
        "num_layers": 3,
        "input_size": 16,
        "hidden_size": 7,
    }
  },
  "iterator": {
    "type": "basic",
    "batch_size": 32
  },
  "trainer": {
    "num_epochs": 10,
    # TODO(brendanr): Switch this to [0, 1].
    "cuda_device" : -1,
    "optimizer": {
      "type": "sgd",
      "lr": 0.01
    }
  }
}
