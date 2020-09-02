import pandas as pd

from contradictory_my_dear_watson.datasets.dataset import Dataset


class ContradictoryMyDearWatsonDataset(Dataset):
    """Dataset class for Contradictory, My Dear Watson dataset."""

    name = 'contradictory-my-dear-watson'

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Dataset specific transformation method.

        Transformation consists of two steps:
            1. Take only English language.
            2. Take only `premise`, `hypothesis` and `label` attributes.

        :param data: data to be transformed.
        :return: transformed data containing string `premise`,
            `hypothesis` and int `label`.
        """
        data = data[data.language == 'English']  # type: ignore
        data = data[['premise', 'hypothesis', 'label']]  # type: ignore
        return data
