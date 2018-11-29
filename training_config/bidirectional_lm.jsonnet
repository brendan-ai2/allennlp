local NUM_GPUS = 2;
# TODO(brendanr): Can be as large as 8 on your machine.
local NUM_THREADS = 8;

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
            "type": "single_id"
          },
          "token_characters": {
            "type": "elmo_characters"
          }
        },
        "max_sequence_length": 500
};

local BASE_ITERATOR = {
  "type": "bucket",
  "max_instances_in_memory": 4096 * NUM_GPUS,
  # TODO(brendanr): How does this interact with maximum_samples_per_batch below?
  "batch_size": 512 * NUM_GPUS,
  # TODO(brendanr): Correct order?
  "sorting_keys": [["source", "num_tokens"]],
  # TODO(brendanr): Is this even meaningful given laziness?
  "biggest_batch_first": true,
  # TODO(brendanr): Grok namespacing vis-a-vis  `["source", "num_tokens"]` above.
  # Notes:
  # - NUM_GPUS * 3k leaves GPUs underutilized.
  # - Same with 6k.
  # - 12k OOMs. (Maybe having a limit on characters per token would help here?
  # - Hmmm, 9k OOMs too. Maybe we have a leak?
  # - 6k OOMs on V100. LEAK?
  "maximum_samples_per_batch": ["num_tokens", NUM_GPUS * 2000]
};

{
  "dataset_reader": if NUM_THREADS > 1 then {
    "type": "multiprocess",
    "base_reader": BASE_READER,
    "num_workers": NUM_THREADS,
    "output_queue_size": 1000
    # TODO(brendanr): Consider epochs_per_read and output_queue_size.
  } else BASE_READER,
  # All data
  #"train_data_path": "/home/brendanr/workbenches/calypso/train/*",
  #"validation_data_path": "/home/brendanr/workbenches/calypso/dev/*",
  # 2 shards for training
  #"train_data_path": "/home/brendanr/workbenches/calypso/train/news.en-0000[2-3]*",
  #"validation_data_path": "/home/brendanr/workbenches/calypso/dev/*",
  # 1 shard for training
  #"train_data_path": "/home/brendanr/workbenches/calypso/train/news.en-00002-of-00100",
  #"validation_data_path": "/home/brendanr/workbenches/calypso/dev/news.en-00001-of-00100",
  # Trivial amount sharded
  #"train_data_path": "/home/brendanr/repos/brendanr/allennlp/allennlp/tests/fixtures/language_modeling/shards/*",
  #"validation_data_path": "/home/brendanr/repos/brendanr/allennlp/allennlp/tests/fixtures/language_modeling/shards/*",
  # Trivial amount sharded -- 2 shards for training
  #"train_data_path": "/home/brendanr/repos/brendanr/allennlp/allennlp/tests/fixtures/language_modeling/shards/shard[0-1]",
  #"validation_data_path": "/home/brendanr/repos/brendanr/allennlp/allennlp/tests/fixtures/language_modeling/shards/shard2",

  # 2 small, but not trivial
  "train_data_path": "/home/brendanr/workbenches/calypso/train_small/*",
  #"validation_data_path": "/home/brendanr/workbenches/calypso/dev_small/*",

  # TODO: Figure out which start and end characters to remove from the tokens.txt file.
  "vocabulary": {
      #"tokens_to_add": {
      #    "tokens": ["<s>", "</s>"],
      #    "token_characters": ["<>/s"]
      #},
      #"min_count": {"source_tokens": 3},
      "directory_path": "/home/brendanr/workbenches/calypso/vocabulary"
  },
  "model": {
    "type": "bidirectional-language-model",
    "num_samples": 8192,
    "sparse_embeddings": true,
    "initializer": [
          [".*tag_projection_layer.*weight", {"type": "xavier_uniform"}], # Initialise the final projection before the softmax.
          [".*tag_projection_layer.*bias", {"type": "zero"}],
          [".*feedforward.*weight", {"type": "xavier_uniform"}], # Same initialisation for an auxilary FF layer. `xavier_uniform` is a better default than random normally.
          [".*feedforward.*bias", {"type": "zero"}],
          # This part is for the LSTM init.
          [".*weight_ih.*", {"type": "xavier_uniform"}],
          [".*weight_hh.*", {"type": "orthogonal"}],
          [".*bias_ih.*", {"type": "zero"}],
          [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
    ],
    "text_field_embedder": {
      # Note: This is because we only use the token_characters during embedding, not the tokens themselves.
      "allow_unmatched_keys": true,
      "token_embedders": {
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
                "num_embeddings": 262,
                # TODO(brendanr): When used with an LSTM contextualizer this is 32. Is that okay?
                "embedding_dim": 16
            },
            "encoder": {
                "type": "cnn-highway",
                "activation": "relu",
                "embedding_dim": 16,
                "filters": [
                    [1, 32],
                    [2, 32],
                    [3, 64],
                    [4, 128],
                    [5, 256],
                    [6, 512],
                    [7, 1024]],
                "num_highway": 2,
                "projection_dim": 512,
                "projection_location": "after_highway",
                # TODO(brendanr): Implement
                #"max_characters_per_token": 50,
                "do_layer_norm": true
            }
        }
      }
    },
    # Applies to the contextualized embeddings.
    "dropout": 0.1,
    # TODO(brendanr): Flesh out. Use Calypso.
    # TODO(brendanr): For any LSTM use Mark's special initialization tricks. Maybe not for transformer.
    "contextualizer": {
        "type": "transformer",
        "input_dim": 512,
        "hidden_dim": 2048,
        "num_layers": 6,
        # TODO(brendanr): Does this need to be used?
        #"dropout": ???,
        # TODO(brendanr): Verify this dropout is applied in the same place as Calypso.
        "input_dropout": 0.1
    }
  },
  #"iterator": BASE_ITERATOR,
  # Note: The multiprocess iterator doesn't make sense with the ShardedDataset model.
  # NOTE: OR DOES IT?
  "iterator": {
    "type": "multiprocess",
    "base_iterator": BASE_ITERATOR,
    "num_workers": NUM_THREADS,
    # Note: Requires a higher ulimit for some reason. I used `ulimit -n 4096`. Could use tuning.
    # TODO(brendanr): Why is this necessary?
    "output_queue_size": 500
  },
  "trainer": {
    "num_epochs": 10,
    "cuda_device" : if NUM_GPUS > 1 then std.range(0, NUM_GPUS - 1) else 0,
    "optimizer": {
      # TODO(brendanr): Use the dense_sparse_adam optimizer.
      # The gradient accumulators in adam for the running stdev and mean for the words that we didn't use are going to drop to 0 if we don't do this, because we would still decay the values to zero, even when we don't use them.
      "type": "dense_sparse_adam"
      # TOO BIG???
      #,"lr": 0.01
    },
    # TODO(brendanr): Needed with transformer too?
    "grad_norm": 10.0,
    "learning_rate_scheduler": {
      "type": "noam",
      # See https://github.com/allenai/calypso/blob/master/calypso/train.py#L401
      "model_size": 512,
      # See https://github.com/allenai/calypso/blob/master/bin/train_transformer_lm1b.py#L51.
      # TODO(brendanr): Adjust based on your sample size vis a vis the Calypso version.
      "warmup_steps": 2000
    }
  }
}
