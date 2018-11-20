local NUM_GPUS = 1;
# TODO(brendanr): Can be as large as 8 on your machine.
local NUM_THREADS = 2;

local BASE_READER = {
        "type": "simple_language_modeling",
        "tokenizer": {
          "type": "word",
          "word_splitter": {
	    # The 1 Billion Word Language Model Benchmark dataset is
	    # pre-tokenized. (Also, if you're running against a untokenized
	    # dataset be aware that there are serialization issues with Spacy.
	    # These come into play in the multiprocess case.)
            "type": "just_spaces"
          }
        },
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
        },
        max_sequence_length: 500,
        maximum_samples_per_batch: NUM_GPUS * 3000
};

{
  "dataset_reader": if NUM_THREADS > 1 then {
    "type": "multiprocess",
    "base_reader": BASE_READER,
    "num_workers": NUM_THREADS,
    "output_queue_size": 100000
    # TODO(brendanr): Consider epochs_per_read and output_queue_size.
  } else BASE_READER,
  # All data
  #"train_data_path": "/home/brendanr/workbenches/calypso/train/*",
  #"validation_data_path": "/home/brendanr/workbenches/calypso/dev/*",
  # 2 shards for training
  "train_data_path": "/home/brendanr/workbenches/calypso/train/news.en-0000[2-3]*",
  "validation_data_path": "/home/brendanr/workbenches/calypso/dev/*",
  # 1 shard for training
  #"train_data_path": "/home/brendanr/workbenches/calypso/train/news.en-00002-of-00100",
  #"validation_data_path": "/home/brendanr/workbenches/calypso/dev/news.en-00001-of-00100",
  # Trivial amount sharded
  #"train_data_path": "/home/brendanr/repos/brendanr/allennlp/allennlp/tests/fixtures/language_modeling/shards/*",
  #"validation_data_path": "/home/brendanr/repos/brendanr/allennlp/allennlp/tests/fixtures/language_modeling/shards/*",
  # Trivial amount sharded -- 2 shards for training
  #"train_data_path": "/home/brendanr/repos/brendanr/allennlp/allennlp/tests/fixtures/language_modeling/shards/shard[0-1]",
  #"validation_data_path": "/home/brendanr/repos/brendanr/allennlp/allennlp/tests/fixtures/language_modeling/shards/shard2",
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
    "type": "bucket",
    "batch_size": 40,
    # TODO(brendanr): Correct order?
    "sorting_keys": [["source", "num_tokens"], ["source", "num_token_characters"]],
    # TODO(brendanr): Is this even meaningful given laziness?
    "biggest_batch_first": true
  },
  #"iterator": {
  #  "type": "multiprocess",
  #  "iterator": {
  #    "type": "basic",
  #    "batch_size": 32
  #  },
  #  "num_workers": 8,
  #  "output_queue_size": 100000
  #},
  "trainer": {
    "num_epochs": 10,
    "cuda_device" : if NUM_GPUS > 1 then std.range(0, NUM_GPUS) else 0,
    "optimizer": {
      "type": "sgd",
      "lr": 0.01
    }
  }
}
