import glob
import itertools
import random
from multiprocessing import Process, Manager, Queue
from typing import Iterable, Iterator, Callable, List
import logging

from allennlp.data.instance import Instance
from allennlp.common import Tqdm
from allennlp.common.checks import ConfigurationError
from allennlp.common.registrable import Registrable

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class _LazyInstances(Iterable):
    """
    An ``Iterable`` that just wraps a thunk for generating instances and calls it for
    each call to ``__iter__``.
    """
    def __init__(self, instance_generator: Callable[[], Iterator[Instance]]) -> None:
        super().__init__()
        self.instance_generator = instance_generator

    def __iter__(self) -> Iterator[Instance]:
        instances = self.instance_generator()
        if isinstance(instances, list):
            raise ConfigurationError("For a lazy dataset reader, _read() must return a generator")
        return instances

class DatasetReader(Registrable):
    """
    A ``DatasetReader`` knows how to turn a file containing a dataset into a collection
    of ``Instance`` s.  To implement your own, just override the `_read(file_path)` method
    to return an ``Iterable`` of the instances. This could be a list containing the instances
    or a lazy generator that returns them one at a time.

    All parameters necessary to _read the data apart from the filepath should be passed
    to the constructor of the ``DatasetReader``.

    Parameters
    ----------
    lazy : ``bool``, optional (default=False)
        If this is true, ``instances()`` will return an object whose ``__iter__`` method
        reloads the dataset each time it's called. Otherwise, ``instances()`` returns a list.
    """
    def __init__(self, lazy: bool = False) -> None:
        self.lazy = lazy

    def read(self, file_path: str) -> Iterable[Instance]:
        """
        Returns an ``Iterable`` containing all the instances
        in the specified dataset.

        If ``self.lazy`` is False, this calls ``self._read()``,
        ensures that the result is a list, then returns the resulting list.

        If ``self.lazy`` is True, this returns an object whose
        ``__iter__`` method calls ``self._read()`` each iteration.
        In this case your implementation of ``_read()`` must also be lazy
        (that is, not load all instances into memory at once), otherwise
        you will get a ``ConfigurationError``.

        In either case, the returned ``Iterable`` can be iterated
        over multiple times. It's unlikely you want to override this function,
        but if you do your result should likewise be repeatedly iterable.
        """
        lazy = getattr(self, 'lazy', None)
        if lazy is None:
            logger.warning("DatasetReader.lazy is not set, "
                           "did you forget to call the superclass constructor?")

        if lazy:
            return _LazyInstances(lambda: iter(self._read(file_path)))
        else:
            instances = self._read(file_path)
            if not isinstance(instances, list):
                instances = [instance for instance in Tqdm.tqdm(instances)]
            if not instances:
                raise ConfigurationError("No instances were read from the given filepath {}. "
                                         "Is the path correct?".format(file_path))
            return instances

    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        Reads the instances from the given file_path and returns them as an
        `Iterable` (which could be a list or could be a generator).
        You are strongly encouraged to use a generator, so that users can
        read a dataset in a lazy way, if they so choose.
        """
        raise NotImplementedError

    def dataset(self, file_path: str) -> 'Dataset':
        return UnshardedDataset(self, file_path)

    def text_to_instance(self, *inputs) -> Instance:
        """
        Does whatever tokenization or processing is necessary to go from textual input to an
        ``Instance``.  The primary intended use for this is with a
        :class:`~allennlp.service.predictors.predictor.Predictor`, which gets text input as a JSON
        object and needs to process it to be input to a model.

        The intent here is to share code between :func:`_read` and what happens at
        model serving time, or any other time you want to make a prediction from new data.  We need
        to process the data in the same way it was done at training time.  Allowing the
        ``DatasetReader`` to process new text lets us accomplish this, as we can just call
        ``DatasetReader.text_to_instance`` when serving predictions.

        The input type here is rather vaguely specified, unfortunately.  The ``Predictor`` will
        have to make some assumptions about the kind of ``DatasetReader`` that it's using, in order
        to pass it the right information.
        """
        raise NotImplementedError

class Dataset:
    def map_partitions(self,
                       f: Callable[[Iterable[Instance]], Iterable]) -> Iterable:
        raise NotImplementedError

    # Warning: Prefer map_partitions if you can structure your problem that way. More parallelism.
    def read(self) -> Iterable[Instance]:
        return self.map_partitions(lambda x: x)

class EmptyDataset(Dataset):
    def map_partitions(self,
                       f: Callable[[Iterable[Instance]], Iterable]) -> Iterable:
        return []

class UnshardedDataset(Dataset):
    def __init__(self, reader: DatasetReader, file_path: str) -> None:
        self._reader = reader
        self._file_path = file_path

    def map_partitions(self,
                       f: Callable[[Iterable[Instance]], Iterable]) -> Iterable:
        iterable = self._reader.read(self._file_path)
        return f(iterable)

class CombinedDataset(Dataset):
    def __init__(self, datasets) -> None:
        self._datasets = datasets

    def map_partitions(self,
                       f: Callable[[Iterable[Instance]], Iterable]) -> Iterable:
        iterables = [dataset.map_partitions(f) for dataset in self._datasets]
        return itertools.chain(iterables)

class Sentinel():
    def __init__(self, id):
        self.id = id

class IterableQueue(Iterable):
    def __init__(self, queue, processes, num_workers, objects_to_retain):
        # Hold a reference to the manager just so it's not garbage collected while we're still iterating.
        # TODO: reconsider
        self._queue = queue
        self._processes = processes
        self._num_workers = num_workers
        # This is simply a way to prevent the objects in the passed list from being garbage collected. Any
        # shared state needed during iteration in the child processes must not be garbage collected in the
        # main process, e.g. input_queue.
        self._objects_to_retain = objects_to_retain

    def __iter__(self) -> Iterator:
        num_finished = 0

        while num_finished < self._num_workers:
            item = self._queue.get()
            if isinstance(item, Sentinel):
                # Means a worker has finished, so increment the finished count.
                num_finished += 1
                logger.info(f"worker {item.id} finished ({num_finished}/{self._num_workers})")
            else:
                # Otherwise it's a real value, so yield it up.
                yield item

        for process in self._processes:
            process.join()
        # TODO: reconsider
        self._processes.clear()

def _worker(f: Callable[[Iterable[Instance]], Iterable],
            reader: DatasetReader,
            input_queue: Queue,
            output_queue: Queue,
            sentinel: Sentinel) -> None:
    # Keep going until you get a file_path that's None.
    while True:
        file_path = input_queue.get()
        if file_path is None:
            # Put the sentinel on the queue to signify that I'm finished
            output_queue.put(sentinel)
            break

        logger.info(f"reading instances from {file_path}")
        iterable = f(reader.read(file_path))
        for element in iterable:
            output_queue.put(element)

class ShardedDataset(Dataset):
    def __init__(self, file_path, reader, num_workers, epochs_per_read, output_queue_size):
        self.file_path = file_path
        self.reader = reader
        self.num_workers = num_workers
        self.epochs_per_read = epochs_per_read
        self.output_queue_size = output_queue_size

    def map_partitions(self,
                       f: Callable[[Iterable[Instance]], Iterable]) -> Iterable:
        shards = glob.glob(self.file_path)
        manager = Manager()

        # If we want multiple epochs per read, put shards in the queue multiple times.
        input_queue = manager.Queue(len(shards) * self.epochs_per_read + self.num_workers)
        for _ in range(self.epochs_per_read):
            random.shuffle(shards)
            for shard in shards:
                input_queue.put(shard)

        # Then put a None per worker to signify no more files.
        for _ in range(self.num_workers):
            input_queue.put(None)

        processes: List[Process] = []
        output_queue = manager.Queue(self.output_queue_size)
        for worker_id in range(self.num_workers):
            process = Process(target=_worker,
                              args=(f, self.reader, input_queue, output_queue, Sentinel(worker_id)))
            logger.info(f"starting worker {worker_id}")
            process.start()
            processes.append(process)

        return IterableQueue(output_queue, processes, self.num_workers, [manager, input_queue])
