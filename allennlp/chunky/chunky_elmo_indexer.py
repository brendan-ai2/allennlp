import itertools
import json
import logging
from typing import Dict, List

import numpy as np
import torch

from overrides import overrides

from allennlp.chunky.chunk_helpers import garble, get_chunks
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import pad_sequence_to_length, prepare_environment
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer, ELMoCharacterMapper
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@TokenIndexer.register("chunky_elmo")
class ChunkyElmoIndexer(TokenIndexer[List[int]]):
    """
    Convert a token to an array of character ids to compute ELMo representations.

    Parameters
    ----------
    namespace : ``str``, optional (default=``elmo_characters``)
    """
    # pylint: disable=no-self-use
    def __init__(self,
                 chunker_path: str,
                 segmental_vocabulary: Vocabulary,
                 preprocessed_chunk_file: str = None,
                 max_span_width: int = 89,
                 update_chunker_params: bool = False,
                 remove_dropout: bool = False,
                 bos_token: str = '<S>',
                 eos_token: str = '</S>',
                 namespace: str = 'chunky_elmo',
                 garble_rate: float = 0.0) -> None:
        self._namespace = namespace
        self._max_span_width = max_span_width
        self._garble_rate = garble_rate

        # First initialize the chunker.
        if preprocessed_chunk_file is not None:
            self.chunks_dict: Dict[str, List[str]] = {}
            self.read_predicted_chunks(preprocessed_chunk_file)
        else:
            self.chunks_dict = None
            logger.info("Reading Chunker from %s", chunker_path)
            from allennlp.models.archival import load_archive
            chunker_archive = load_archive(chunker_path)
            self.chunker = chunker_archive.model

            if not update_chunker_params:
                for param in self.chunker.parameters():
                    param.requires_grad_(False)

            if remove_dropout:
                # Setting dropout to 0.0 for all parameters in chunker.
                self.chunker.dropout.p = 0.0
                self.chunker.encoder._module.dropout = 0.0
                self.chunker.text_field_embedder.token_embedder_elmo._elmo._dropout.p =0.0

        self.elmo_indexer = ELMoTokenCharactersIndexer(namespace='elmo_characters')
        self.token_indexer = SingleIdTokenIndexer()

        self.seglm_vocab = segmental_vocabulary #load_archive(segmental_path).model.vocab
        self.bos_token = bos_token
        self.eos_token = eos_token

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        pass

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[List[int]]]:
        chunk_tags = self.get_chunk_tags(tokens, vocabulary)

        # Add BOS, EOS characters
        tokens_with_bos_eos = [Token(self.bos_token)] + tokens + [Token(self.bos_token)]
        character_indices_with_eos_bos = self.elmo_indexer.tokens_to_indices(tokens_with_bos_eos, vocabulary, "elmo")

        # Get string chunk tags.
        chunk_tags_str, instance_fields = self.get_input_data_structures_for_segmental_lm(chunk_tags)
        # Convert these into tags for the language model.
        chunk_tags_seglm_ids = self.get_tags_in_lm_vocab(chunk_tags_str)

        return_dict = {'character_ids': character_indices_with_eos_bos["elmo"],
                       'mask': [1] * len(tokens),
                       "mask_with_bos_eos": [1] * len(tokens_with_bos_eos),
                       'tags': chunk_tags_seglm_ids}
        return_dict.update(instance_fields)

        return return_dict

    @overrides
    def get_padding_token(self) -> int:
        # TODO(Swabha/Matt): Exact replica of `openai_transformer_byte_pair_indexer`.
        return 0

    @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:  # pylint: disable=unused-argument
        return {}

    @staticmethod
    def _default_value_for_character_id_padding():
        return [0] * ELMoCharacterMapper.max_word_length

    @staticmethod
    def _default_value_for_indices():
        return -1

    @staticmethod
    def _default_value_for_mask():
        return 0

    @overrides
    def pad_token_sequence(self,
                           tokens: Dict[str, List[int]],
                           desired_num_tokens: Dict[str, int],
                           padding_lengths: Dict[str, int]) -> Dict[str, List[int]]:  # pylint: disable=unused-argument
        ret = {}
        for key, default_val in [
            ['character_ids', self._default_value_for_character_id_padding],
            ['seg_ends', self._default_value_for_indices],
            ['seg_starts', self._default_value_for_indices],
            ['seg_map', self._default_value_for_indices],
            ['tags', self._default_value_for_mask],
            ['mask', self._default_value_for_mask],
            ['mask_with_bos_eos', self._default_value_for_mask]
        ]:
            ret[key] = pad_sequence_to_length(tokens[key],
                                              desired_num_tokens[key],
                                              default_value=default_val)
        return ret

    def get_input_data_structures_for_segmental_lm(self,
                                                   chunk_tags: List[str]):
        """
        Logic from SegmentalConll2000DatasetReader
        """
        chunk_tags = ['U-O' if tag == 'O' else tag for tag in chunk_tags]

        if np.random.uniform() < self._garble_rate:
            chunk_tags = garble(chunk_tags)

        # Add BOS-EOS tags
        chunk_tags = ['U-O'] + chunk_tags + ['U-O']

        instance_fields = {}
        seg_starts = []
        seg_ends = []
        seg_map = []
        seg_count = 0
        for i, tag in enumerate(chunk_tags):
            if tag.startswith('B-') or tag.startswith('U-'):
                start = i
                seg_starts.append(start)
            if tag.startswith('L-') or tag.startswith('U-'):
                end = i
                assert end - start < self._max_span_width
                seg_ends.append(end)
                seg_map += [seg_count for _ in range(start, end+1)]
                seg_count += 1

        instance_fields['seg_ends'] = seg_ends
        instance_fields['seg_starts'] = seg_starts
        instance_fields['seg_map'] = seg_map

        return chunk_tags, instance_fields

    def get_tags_in_lm_vocab(self, chunk_tags_str:str):
        return [self.seglm_vocab.get_token_index(t, "labels") for t in chunk_tags_str]

    def read_predicted_chunks(self, preprocessed_chunk_file: str):
        tot = acc = 0.
        for line in open(preprocessed_chunk_file, "r"):
            cdict = json.loads(line)
            key = " ".join(cdict["words"])
            if key in self.chunks_dict:
                old_tags = self.chunks_dict[key]
                new_tags = cdict["tags"]
                if old_tags == new_tags:
                    acc += 1
                tot += 1
            self.chunks_dict[key] = cdict["tags"]
        logger.info("Chunk Tag Consistency: %f (%d/%d)", acc/tot, acc, tot)

    def get_chunk_tags(self, tokens: List[Token], vocabulary: Vocabulary):
        if self.chunks_dict is not None:
            sentence = ' '.join([token.text for token in tokens])
            if sentence not in self.chunks_dict:
                raise ConfigurationError(f"Sentence is not in the dictionary: {sentence}")
            return self.chunks_dict[sentence]

        character_indices = self.elmo_indexer.tokens_to_indices(tokens, vocabulary, "elmo")

         # TODO(Swabha/Matt): worry about cuda, cudifying the model and constructor
        character_indices_tensor = {"elmo": torch.LongTensor(character_indices["elmo"]).unsqueeze(0)}

         # The chunker model is a CRF tagger (allennlp.models.crf_tagger)
        output_dict = self.chunker(character_indices_tensor)
        chunk_tag_ids = output_dict["tags"][0]
        chunk_tags = [self.chunker.vocab._index_to_token["labels"][tag] for tag in chunk_tag_ids]
        with open('tmp.json', 'a', encoding='utf-8') as output_json:
            json.dump({"words": [token.text for token in tokens], "tags": chunk_tags}, output_json)
            output_json.write("\n")
        return chunk_tags


if __name__== "__main__":
    from calypso.labeled_seglm_transformer import LabeledSegLMTransformer
    self = ChunkyElmoIndexer("/Users/swabhas/pretrained/log_chunking_ptb_comparable/model.tar.gz",
        "/Users/swabhas/pretrained/log_1b_labeled_seglm_transformer/model.tar.gz")
    sentence = "I have a good dog ."
    tokens =[Token(word) for word in sentence.split()]
    vocabulary = Vocabulary()
    index_name = "test-chunky"

    batch = ["I have a good dog .", "He fetches me stuff ."]

    from allennlp.data.dataset import Batch, Instance
    from allennlp.data.fields.text_field import TextField
    instances = []
    for sentence in batch:
        tokens = [Token(token) for token in sentence]
        field = TextField(tokens, {'test_chunky': self})
        instance = Instance({"elmo_chunky": field})
        instances.append(instance)

    dataset = Batch(instances)
    vocab = Vocabulary()
    dataset.index_instances(vocab)

