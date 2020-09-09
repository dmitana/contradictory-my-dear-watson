from torch import Tensor

from contradictory_my_dear_watson.metrics.metric import Metric


class Accuracy(Metric):
    """
    Accuracy metric class.

    :param self.num_samples: int, number of samples processed.
    :param self.num_samples_prev: int, number of samples processed in
        previous `self.compute()`.
    :param self.num_correct: int, accumulated value of correct hits.
        Value must be reseted after `self.compute()` call.
    """

    def __init__(self):
        """Create a new instance of `Accuracy`."""
        super().__init__()
        self._num_samples = 0
        self._num_samples_prev = 0
        self._num_correct = 0

    @property
    def num_samples(self) -> int:
        """
        Return property `num_samples`.

        :return: number of samples processed.
        """
        return self._num_samples

    @property
    def num_samples_prev(self) -> int:
        """
        Return property `num_samples_prev`.

        :return: number of samples processed in previous
            `self.compute()`.
        """
        return self._num_samples_prev

    @property
    def num_correct(self) -> int:
        """
        Return property `num_correct`.

        :return: accumulated value of correct hits.
        """
        return self._num_correct

    def update(self, output: Tensor, label: Tensor) -> 'Accuracy':
        """
        Update metric variables.

        `self.num_correct` is updated based on `output` and `label` and
        `self.num_samples` is incremented by length of output.

        Method must be called before `self.compute()` method. When
        `self.update()` and `self.compute()` are called in a loop, moving
        average is computed.

        :param output: model output.
        :param label: gold label.
        :return: self object.
        """
        pred = output.argmax(dim=1, keepdim=True)
        self._num_correct += pred.eq(label.view_as(pred)).sum().item()
        self._num_samples += len(output)

        return self

    def compute(self) -> 'Accuracy':
        """
        Compute final metric value.

        Set `self.result` and `self.num_samples_prev` property and also
        reset `self.num_correct` to 0.

        When `self.update()` and `self.compute()` are called in a loop,
        moving average is computed.

        :return: self object.
        """
        if self.first:
            self._result = self.num_correct / self.num_samples
            self._first = False
        else:
            self._result = (self.num_correct + self.num_samples_prev *
                            self.result) / self.num_samples

        self._num_samples_prev = self._num_samples
        self._num_correct = 0

        return self
