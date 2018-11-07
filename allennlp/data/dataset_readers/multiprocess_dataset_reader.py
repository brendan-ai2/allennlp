from typing import List, Iterable, Iterator, Callable, Any
import glob
import logging
import random

from torch.multiprocessing import Manager, Process, Queue, log_to_stderr

from allennlp.data.dataset import Dataset, ShardedDataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance

logger = log_to_stderr()  # pylint: disable=invalid-name
logger.setLevel(logging.INFO)


@DatasetReader.register('multiprocess')
class MultiprocessDatasetReader(DatasetReader):
    """
    Wraps another dataset reader and uses it to read from multiple input files
    using multiple processes. Note that in this case the ``file_path`` passed to ``read()``
    should be a glob, and that the dataset reader will return instances from all files
    matching the glob.

    Parameters
    ----------
    base_reader : ``DatasetReader``
        Each process will use this dataset reader to read zero or more files.
    num_workers : ``int``
        How many data-reading processes to run simultaneously.
    epochs_per_read : ``int``, (optional, default=1)
        Normally a call to ``DatasetReader.read()`` returns a single epoch worth of instances,
        and your ``DataIterator`` handles iteration over multiple epochs. However, in the
        multiple-process case, it's possible that you'd want finished workers to continue on to the
        next epoch even while others are still finishing the previous epoch. Passing in a value
        larger than 1 allows that to happen.
    output_queue_size: ``int``, (optional, default=1000)
        The size of the queue on which read instances are placed to be yielded.
        You might need to increase this if you're generating instances too quickly.
    """
    def __init__(self,
                 base_reader: DatasetReader,
                 num_workers: int,
                 epochs_per_read: int = 1,
                 # TODO: Warning about queue size serving multiple purposes effectively.
                 output_queue_size: int = 1000) -> None:
        # Multiprocess reader is intrinsically lazy.
        super().__init__(lazy=True)

        self.reader = base_reader
        self.num_workers = num_workers
        self.epochs_per_read = epochs_per_read
        self.output_queue_size = output_queue_size

    def text_to_instance(self, *args, **kwargs) -> Instance:
        """
        Just delegate to the base reader text_to_instance.
        """
        # pylint: disable=arguments-differ
        return self.reader.text_to_instance(*args, **kwargs)

    def _read(self, file_path: str) -> Iterable[Instance]:
        raise RuntimeError("Multiprocess reader implements read() directly.")

    def read(self, file_path: str) -> Iterable[Instance]:
        raise Exception("Don't call this!!")
        dataset = self.dataset(file_path)
        return dataset.read()

    def dataset(self, file_path: str) -> Dataset:
        return ShardedDataset(file_path, self.reader, self.num_workers, self.epochs_per_read, self.output_queue_size)
