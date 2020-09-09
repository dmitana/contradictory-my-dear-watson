from contradictory_my_dear_watson.metrics.metric import Metric


class AvgLoss(Metric):
    """
    Average Loss "metric".

    :param self.num_batches: int, number of batches processed.
    :param self.loss: float, accumulated loss value. Value must be
        reseted after `self.compute()` call.
    """

    def __init__(self):
        """Create a new instance of `AvgLoss`."""
        super().__init__()
        self._num_batches = 0
        self._loss = 0.0

    @property
    def num_batches(self) -> int:
        """
        Return property `num_batches`.

        :return: number of batches processed.
        """
        return self._num_batches

    @property
    def loss(self) -> float:
        """
        Return property `loss`.

        :return: accumulated loss value.
        """
        return self._loss

    def update(self, loss: float) -> 'AvgLoss':
        """
        Update metric variables.

        `loss` is added to `self.loss` and `self.num_batches` is
        incremented by 1.

        Method must be called before `self.compute()` method. When
        `self.update()` and `self.compute()` are called in a loop, moving
        average is computed.

        :param loss: loss to be added.
        :return: self object.
        """
        self._loss += loss
        self._num_batches += 1

        return self

    def compute(self) -> 'AvgLoss':
        """
        Compute final metric value.

        Set `self.result` property and reset `self.loss` to 0.0.

        When `self.update()` and `self.compute()` are called in a loop,
        moving average is computed.

        :return: self object.
        """
        if self.first:
            self._result = self.loss / self.num_batches
            self._first = False
        else:
            self._result = (self.loss + (self.num_batches - 1) *
                            self.result) / self.num_batches

        self._loss = 0.0

        return self
