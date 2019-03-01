# pylint: disable=no-self-use,invalid-name
import numpy as np

from allennlp.common.testing import AllenNlpTestCase
from allennlp.chunky.chunk_helpers import get_chunks, make, grow, shrink, garble_helper, switch_tag, garble


class TestChunkyElmoIndexer(AllenNlpTestCase):
    def test_get_chunks(self):
        tags = ['U-O', 'U-NP', 'U-VP', 'B-NP', 'I-NP', 'L-NP', 'U-PP', 'B-NP', 'L-NP', 'U-ADVP',
                'U-SBAR', 'B-NP', 'L-NP', 'U-NP', 'U-VP', 'U-ADJP', 'U-PP', 'U-O', 'U-O']
        chunks = get_chunks(tags)
        expected = [['U-O'], ['U-NP'], ['U-VP'], ['B-NP', 'I-NP', 'L-NP'], ['U-PP'],
                    ['B-NP', 'L-NP'], ['U-ADVP'], ['U-SBAR'], ['B-NP', 'L-NP'], ['U-NP'],
                    ['U-VP'], ['U-ADJP'], ['U-PP'], ['U-O'], ['U-O']]
        assert chunks == expected

    def test_make(self):
        assert make(1, 'FOO') == ['U-FOO']
        assert make(2, 'FOO') == ['B-FOO', 'L-FOO']
        assert make(3, 'FOO') == ['B-FOO', 'I-FOO', 'L-FOO']

    def test_grow(self):
        assert grow(['U-FOO']) == ['B-FOO', 'L-FOO']

    def test_shrink(self):
        assert shrink(['B-FOO', 'L-FOO']) == ['U-FOO']
        assert shrink(['U-FOO']) == []

    def test_garble_helper(self):
        tags = ['U-O', 'U-NP', 'U-VP', 'B-NP', 'I-NP', 'L-NP', 'U-PP', 'B-NP', 'L-NP']
        chunks = get_chunks(tags)
        garble_helper(chunks, 2, 0)
        expected = [['U-O'], ['U-NP'], ['B-VP', 'L-VP'], ['B-NP', 'L-NP'], ['U-PP'], ['B-NP', 'L-NP']]
        assert chunks == expected
        garble_helper(chunks, 5, 0)
        print(chunks)
        expected2 = [['U-O'], ['U-NP'], ['B-VP', 'L-VP'], ['B-NP', 'L-NP'], [], ['B-NP', 'I-NP', 'L-NP']]
        assert chunks == expected2

    def test_switch_tags(self):
        np.random.seed(173)
        tags = ['U-O', 'U-NP', 'U-VP', 'B-NP', 'I-NP', 'L-NP', 'U-PP', 'B-NP', 'L-NP']
        chunks = get_chunks(tags)
        switch_tag(chunks, 3)
        expected = [['U-O'], ['U-NP'], ['U-VP'], ['B-ADJP', 'I-ADJP', 'L-ADJP'], ['U-PP'], ['B-NP', 'L-NP']]
        assert chunks == expected

    def test_garble(self):
        np.random.seed(442)
        tags = ['U-O', 'U-NP', 'U-VP', 'B-NP', 'I-NP', 'L-NP', 'U-PP', 'B-NP', 'L-NP']
        expected = ['U-O', 'U-NP', 'B-VP', 'L-VP', 'B-NP', 'L-NP', 'U-PP', 'B-SBAR', 'L-SBAR']
        assert garble(tags) == expected
