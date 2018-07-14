from typing import Optional

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric


@Metric.register("sequence_accuracy")
class SequenceAccuracy(Metric):
    """
    Sequence Top-K accuracy. Assumes integer labels, with
    each item to be classified having a single correct class.
    """
    def __init__(self, top_k: int = 1) -> None:
        self._top_k = top_k
        self.correct_count = 0.0
        self.total_count = 0.0

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, top_k, sequence_length).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, sequence_length).
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        # Some sanity checks.
        num_classes = predictions.size(-1)
        if gold_labels.dim() != predictions.dim() - 1:
            raise ConfigurationError("gold_labels must have dimension == predictions.size() - 1 but "
                                     "found tensor of shape: {}".format(predictions.size()))

        # gold = torch.tensor([[1, 2, 3], [2, 4, 8], [0, 1, 1]])
        # guesses = torch.tensor([[[1, 2, 3], [1, 2, -1]], [[2, 4, 8], [2, 5, 9]], [[-1, -1, -1], [0, 1, 1]]])
        # gold2 = gold.unsqueeze(1)
        k = predictions.size()[1]
        print("k: {}".format(k))
        print("gold_labels.size(): {}".format(gold_labels.size()))
        expanded_size = list(gold_labels.size())
        expanded_size.insert(1, k)
        print("expanded_size: {}".format(expanded_size))
        expanded_gold = gold_labels.unsqueeze(1).expand(expanded_size)
        eqs = expanded_gold.eq(predictions)
        matches_per_question = eqs.min(dim=2)[0]
        some_match = matches_per_question.max(dim=1)[0]
        correct = some_match.sum().item()
        # TODO(brendanr): Handle mask.

        ## Top K indexes of the predictions (or fewer, if there aren't K of them).
        ## Special case topk == 1, because it's common and .max() is much faster than .topk().
        #if self._top_k == 1:
        #    top_k = predictions.max(-1)[1].unsqueeze(-1)
        #else:
        #    top_k = predictions.topk(min(self._top_k, predictions.shape[-1]), -1)[1]

        ## This is of shape (batch_size, ..., top_k).
        #correct = top_k.eq(gold_labels.long().unsqueeze(-1)).float()

        if mask is not None:
            #correct *= mask.float().unsqueeze(-1)
            #self.total_count += mask.sum()
            pass
        else:
            self.total_count += predictions.size()[0]
        self.correct_count += correct

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        accuracy = float(self.correct_count) / float(self.total_count)
        if reset:
            self.reset()
        return accuracy

    @overrides
    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0
