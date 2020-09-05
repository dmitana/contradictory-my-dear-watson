from abc import ABC, abstractmethod
import logging
import os
from typing import Any

import numpy as np
import pandas as pd


class Dataset(ABC):
    """
    Dataset abstract class.

    :param label_map: dict, map labels to numbers.
    :param self.data_path: str, path to the data.
    :param self.train_size: int, size of train dataset in percentage.
    :param self.dev_size: int, size of development dataset in percentage.
    :param self.test_size: int, size of test dataset in percentage.
    :param self.sanity_size: int, if greater than `0` then create dataset
        for sanity check with `--sanity-size` elements. Ignore
        `train_size`, `dev_size` and `test_size` if set.'
    """

    label_map = {
        'entailment': 0,
        'neutral': 1,
        'contradiction': 2
    }

    def __init__(
        self,
        data_path: str,
        train_size: int,
        dev_size: int,
        test_size: int,
        sanity_size: int = 0,
    ):
        """
        Create a new instance of `ContradictoryMyDearWatsonDataset`.

        :param data_path: path to the data.
        :param train_size: size of train dataset in percentage.
        :param dev_size: size of development dataset in percentage.
        :param test_size: size of test dataset in percentage.
        :param sanity_size: if greater than `0` then create dataset for
            sanity check with `--sanity-size` elements. Ignore
            `train_size`, `dev_size` and `test_size` if set.'
        :raise:
            ValueError: when wrong combination of values are assigned to
                `train_size`, `dev_size`, `test_size` and `sanity_size`.
        """
        self._logger = logging.getLogger(__name__) \
            .getChild(self.__class__.__name__)

        self._data_path = data_path
        self._train_size = train_size
        self._dev_size = dev_size
        self._test_size = test_size
        self._sanity_size = sanity_size

        if not self.check_datasets_sizes():
            raise ValueError(
                f'Unprocessable values of `train_size={self.train_size}`, '
                f'`dev_size={self.dev_size}`, `test_size={test_size}` and '
                f'`sanity_size={self.sanity_size}`.'
            )

    @property
    def data_path(self) -> str:
        """
        Return property `data_path`.

        :return: path to the data.
        """
        return self._data_path

    @property
    def train_size(self) -> int:
        """
        Return property `train_size`.

        :return: size of train dataset in percentage.
        """
        return self._train_size

    @property
    def dev_size(self) -> int:
        """
        Return property `dev_size`.

        :return: size of development dataset in percentage.
        """
        return self._dev_size

    @property
    def test_size(self) -> int:
        """
        Return property `test_size`.

        :return: size of test dataset in percentage.
        """
        return self._test_size

    @property
    def sanity_size(self) -> int:
        """
        Return property `sanity_size`.

        :return: size of sanity check dataset in number of samples.
        """
        return self._sanity_size

    @property
    def name(self) -> str:  # noqa: D102
        raise NotImplementedError(
            'Subclass must define attribute string `name` representing '
            'dataset name.'
        )

    def check_datasets_sizes(self) -> bool:
        """
        Check that given dataset sizes are meaningful.

        :return: `True` if dataset sizes are meaningful, otherwise
            `False`.
        """
        if self.sanity_size == 0:
            if self.train_size < 0 or self.dev_size < 0 or self.test_size < 0:
                return False
            elif self.train_size + self.dev_size + self.test_size > 100:
                return False
            elif self.train_size + self.dev_size + self.test_size == 0:
                return False
        return True

    @abstractmethod
    def read_data(self) -> Any:
        """
        Read data specific method.

        :return: read data.
        """
        pass

    @abstractmethod
    def transform(self, data: Any) -> pd.DataFrame:  # noqa: U100
        """
        Dataset specific transformation method.

        :param data: data to be transformed.
        :return: transformed data containing string `premise`,
            `hypothesis` and int `label`.
        """
        pass

    def __call__(self) -> None:
        """Create datasets."""
        # Read data
        data = self.read_data()

        # Transform data
        dataset = self.transform(data)

        # TODO: add sanity check dataset

        # Get random permutation
        dataset_len = len(dataset)
        np.random.seed(13)  # type: ignore
        np.random.permutation(dataset_len)  # type: ignore

        # Compute lengths of dataset
        train_len = int(dataset_len * self.train_size / 100)
        dev_len = train_len + int(dataset_len * self.dev_size / 100)

        # Create datasets
        train_dataset = dataset[:train_len]
        dev_dataset = dataset[train_len:dev_len]
        test_dataset = dataset[dev_len:]
        assert isinstance(train_dataset, pd.DataFrame)
        assert isinstance(dev_dataset, pd.DataFrame)
        assert isinstance(test_dataset, pd.DataFrame)

        datasets = {
            'train': train_dataset,
            'dev': dev_dataset,
            'test': test_dataset
        }

        dataset_path = f'data/processed/{self.name}'
        os.makedirs(dataset_path, exist_ok=True)

        for key, value in datasets.items():
            length = len(value)
            if length > 0:
                path = os.path.join(dataset_path, f'{key}.csv')
                self._logger.info(
                    f'{key.capitalize()} dataset of length `{len(value)}` '
                    f'samples saving to `{path}`'
                )
                value.to_csv(path, index=False)
