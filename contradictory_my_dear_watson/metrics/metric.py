from abc import ABC, abstractmethod
import logging


class Metric(ABC):
    """
    Metric abstract class.

    :param self.first: bool, whether `self.compute()` is called for the
        first time.
    :param self.result: float, final metric value. Set after
        self.compute() call.
    """

    def __init__(self):
        """Abstract constructor for `Metric` sub-classes."""
        self._logger = logging.getLogger(__name__) \
            .getChild(self.__class__.__name__)

        self._first = True
        self._result = 0.0

    @property
    def first(self) -> bool:
        """
        Return property `first`.

        :return: whether `self.compute()` is called for the first time.
        """
        return self._first

    @property
    def result(self) -> float:
        """
        Return property `result`.

        :return: final metric value.
        """
        return self._result

    @abstractmethod
    def update(self, *args) -> 'Metric':  # noqa: U100
        """
        Update metric variables.

        Method must be called before `self.compute()` method. When
        `self.update()` and `self.compute()` are called in a loop, moving
        average is computed.

        :param *args: metric specific values.
        :return: self object.
        """
        pass

    @abstractmethod
    def compute(self) -> 'Metric':
        """
        Compute final metric value.

        Set `self.result` property.

        When `self.update()` and `self.compute()` are called in a loop,
        moving average is computed.

        :return: self object.
        """
        pass
