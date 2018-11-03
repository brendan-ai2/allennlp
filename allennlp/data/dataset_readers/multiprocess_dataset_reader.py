from typing import List, Iterable, Iterator, Callable, Any
import glob
import logging
import random

from torch.multiprocessing import Manager, Process, Queue, log_to_stderr

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance

logger = log_to_stderr()  # pylint: disable=invalid-name
logger.setLevel(logging.INFO)

def _worker(task: Callable[[Iterable[Instance], Queue], None],
            reader: DatasetReader,
            input_queue: Queue,
            output_queue: Queue,
            index: int) -> None:
    # Keep going until you get a file_path that's None.
    while True:
        file_path = input_queue.get()
        if file_path is None:
            logger.info(f"worker {index} finished")
            break

        logger.info(f"reading instances from {file_path}")
        task(reader.read(file_path), output_queue)


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
        outer_self = self

        class Dataset:
            def __init__(self) :
                self.num_workers = outer_self.num_workers
                self.manager = Manager()

                shards = glob.glob(file_path)
                num_shards = len(shards)

                # If we want multiple epochs per read, put shards in the queue multiple times.
                self.input_queue = self.manager.Queue(num_shards * self.epochs_per_read + self.num_workers)
                for _ in range(self.epochs_per_read):
                    random.shuffle(shards)
                    for shard in shards:
                        self.input_queue.put(shard)

                # Then put a None per worker to signify no more files.
                for _ in range(self.num_workers):
                    self.input_queue.put(None)

            def do(self,
                   task: Callable[[Iterable[Instance], Queue], None],
                   merger: Callable[[Queue], Any]) -> Any:
                processes: List[Process] = []

                output_queue = self.manager.Queue(outer_self.output_queue_size)
                for worker_id in range(self.num_workers):
                    process = Process(target=_worker,
                                      args=(task, outer_self.reader, self.input_queue, output_queue, worker_id))
                    logger.info(f"starting worker {worker_id}")
                    process.start()
                    processes.append(process)

                for process in processes:
                    process.join()
                processes.clear()

                # TODO: There should be a way to consume from the queue in a multiprocess setting, I think.
                return merger(output_queue)

            # TODO(brendanr): Define __iter__ in terms of do? Basically just have a no-op merger and yield up
            # everthing using the strategy Joel used. Might need an extra class and some kind of wrapper for the output
            # objects to detect when everything is finished.
            #def __iter__(self) -> Iterator[Instance]:

        return Dataset()
