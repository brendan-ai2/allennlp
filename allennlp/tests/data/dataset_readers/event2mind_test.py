# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.data.dataset_readers import Event2MindDatasetReader
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase

class TestEvent2MindDatasetReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_default_format(self, lazy):
        reader = Event2MindDatasetReader(lazy=lazy)
        instances = reader.read(str(AllenNlpTestCase.FIXTURES_ROOT / 'data' / 'event2mind.csv'))
        instances = ensure_list(instances)

        for i in instances:
            print("{}\n".format(str(i)))
        # TODO(brendanr): Testify!
        assert(False)
