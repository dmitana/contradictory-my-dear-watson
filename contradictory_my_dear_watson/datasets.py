from typing import Any, Dict, Optional

import pandas as pd
from torch.utils.data import Dataset
from torchtext.vocab import Vocab
from torchvision.transforms import Compose


class NLIDataset(Dataset):
    """
    Natural language inference dataset.

    :param self.data: pandas.DataFrame, NLI data containing `premise`,
        `hypothesis` and `label`.
    :param self.vocab: torchtext.vocab.Vocab, vocabulary created from
        training data.
    :param self.transforms: torchvision.transforms.Compose, composition
        of transformations.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        vocab: Vocab,
        transforms: Optional[Compose] = None
    ):
        """
        Create a new instance of `NLIDataset`.

        :param data: NLI data containing `premise`, `hypothesis` and
            `label`.
        :param vocab: vocabulary created from training data.
        :param transforms: composition of transformations.
        """
        self._data = data
        self._vocab = vocab
        self._transforms = transforms

    @property
    def data(self) -> pd.DataFrame:
        """
        Return property `data`.

        :return: NLI data containing `premise`, `hypothesis` and `label`.
        """
        return self._data

    @property
    def vocab(self) -> Vocab:
        """
        Return property `vocab`.

        :return: vocabulary created from training data.
        """
        return self._vocab

    @property
    def transforms(self) -> Optional[Compose]:
        """
        Return property `transforms`.

        :return: composition of transformations.
        """
        return self._transforms

    def __getitem__(self, i: int) -> Dict[str, Any]:
        """
        Get sample by index `i`.

        If `self.transforms` is not `None`, then returned `sample` is
        preprocessed using this transformations.

        :param i: index of sample to get.
        :return: sample containing `premise`, `hypothesis` and `label`.
        """
        item: Any = self.data.iloc[i]
        sample = {
            'premise': item.premise,
            'hypothesis': item.hypothesis,
            'label': item.label
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """
        Return length of a dataset.

        :return: length of a dataset.
        """
        return len(self.data)
