import json
from typing import Iterator

import pandas as pd
from tqdm import tqdm

from contradictory_my_dear_watson.datasets.dataset import Dataset


class SNLI(Dataset):
    """Dataset class for SNLI dataset."""

    name = 'snli'

    def read_data(self) -> Iterator[str]:
        """
        Read data specific method.

        Read JSONL file line by line.

        :return: line generator.
        """
        with open(self.data_path, mode='r') as fr:
            for line in fr:
                yield json.loads(line)

    def transform(self, data: Iterator[str]) -> pd.DataFrame:
        """
        Dataset specific transformation method.

        Transformation consists of these steps:
            1. Take only `gold_label` equals to `entailment`,
                `contradictory` or `neutral` and map it to appropriate
                number value using `self.label_map`.
            2. Map `sentence1` as `premise`.
            3. Map `sentence2` as 'hypothesis`.

        :param data: data to be transformed.
        :return: transformed data containing string `premise`,
            `hypothesis` and int `label`.
        """
        premises, hypotheses, labels = [], [], []

        self._logger.info('Transforming data ...')
        for sample in tqdm(data):
            gold_label = sample['gold_label']
            premise = sample['sentence1']
            hypothesis = sample['sentence2']
            if gold_label in self.label_map and hypothesis.lower() != 'n/a':
                premises.append(premise)
                hypotheses.append(hypothesis)
                labels.append(self.label_map[gold_label])

        return pd.DataFrame({
            'premise': premises,
            'hypothesis': hypotheses,
            'label': labels
        })
