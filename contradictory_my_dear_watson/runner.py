from functools import partial
import logging
import os
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchvision.transforms import Compose
from transformers import AutoTokenizer

from contradictory_my_dear_watson.datasets import NLIDataset
from contradictory_my_dear_watson.models import BiLSTMModel, Transformer
from contradictory_my_dear_watson.test import test_fn
from contradictory_my_dear_watson.train import train_fn
from contradictory_my_dear_watson.transforms import (
    ToTensor,
    ToVocabulary,
    Tokenize
)
from contradictory_my_dear_watson.utils.dataloader import batch_fn


class Runner:
    """
    Class for running training and evaluation.

    :param self.train_data_path: str, path to the training dataset.
    :param self.test_data_path: Optional[str], path to the development or
        test dataset in train or evaluation mode, respectively.
    :param self.models_dir: Optional[str], path to the directory where
        models are saved each epoch. If `None` then models will not be
        saved.
    :param self.logs_dir: Optional[str], path to the directory where
        TensorBoard logs are saved each epoch. If `None` then logs will
        not be saved.
    :param self.checkpoint_path: Optional[str], path to the model
        checkpoint to be evaluated or to continue in a training from it.
    :param self.num_workers: int, how many subprocesses to use for data
        loading. `0` means that the data will be loaded in the main
        process.
    :param self.print_summary: bool. whether to print model summary.
    :param self.hparams: dict, dictionary with hyperparameters for model
        to be trained.
    :param self.train_data: pd.DataFrame, loaded train data.
    :param self.test_data: Optional[pd.DataFrame], loaded test data.
    :param self.train_dataloader: DataLoader, train dataloader.
    :param self.test_dataloader: Optional[DataLoader], test dataloader.
    :param self.model: nn.Module, model to be trained.
    :param self.device: torch.device, device to be model trained on.
    :param self.optimizer: Optimizer, optimization algorithm.
    :param self.criterion: _Loss, loss function.
    """

    def __init__(
        self,
        model_class: nn.Module,
        train_data_path: str,
        test_data_path: Optional[str] = None,
        models_dir: Optional[str] = None,
        logs_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        num_workers: int = 0,
        print_summary: bool = False,
        **hparams: Dict[str, Any]
    ):
        """
        Create a new instance of `Runner`.

        :param model_class: class of a model to be trained.
        :param train_data_path: path to the training dataset.
        :param test_data_path: path to the development or test dataset
            in train or evaluation mode, respectively.
        :param models_dir: path to the directory where models are saved
            each epoch. If `None` then models will not be saved.
        :param logs_dir: path to the directory where TensorBoard logs are
            saved each epoch. If `None` then logs will not be saved.
        :param checkpoint_path: path to the model checkpoint to be
            evaluated or to continue in a training from it.
        :param num_workers: how many subprocesses to use for data
            loading. `0` means that the data will be loaded in the main
            process.
        :param print_summary: checkpoint_path
        :param **hparams: dictionary with hyperparameters for model to be
            trained. Possible hyperparameters:
                Shared hyperparameters:
                    learning_rate: float, speed rate of a learning.
                    batch_size: int, size of one mini-batch.
                    epochs: int number of training epochs.
                    dropout_prob: float, probability of an element to be
                        zeroed.

                    BiLSTM hyperparameters:
                        lstm_hidden_size: int, number of neurons in a
                            LSTM layer.
                        max_pooling: bool, whether max-pooling over
                            LSTM's output is enabled or not. If
                            max-pooling is not enabled, only last LSTM's
                            hidden states are used.
                        tokenizer: str, tokenizer name to be used by
                            PyTorch `get_tokenizer` function to get
                            instance of `Tokenizer`.
                        tokenizer_language: str, tokenizer language to be
                            used by PyTorch `get_tokenizer` function to
                            get instance of `Tokenizer`.
                        vectors: str, vectors name to be loaded by
                            PyTorch `load_vectors` method.

                    Transformer hyperparameters:
                        pretrained_transformer: str, model name or path
                            of pretrained Transformer model.
        """
        self._logger = logging.getLogger(__name__) \
            .getChild(self.__class__.__name__)

        self._model_class = model_class
        self._train_data_path = train_data_path
        self._test_data_path = test_data_path
        self._models_dir = models_dir
        self._logs_dir = logs_dir
        self._checkpoint_path = checkpoint_path
        self._num_workers = num_workers
        self._print_summary = print_summary
        self._hparams = hparams

        # Load data
        self._logger.info(f'Loading train data `{self.train_data_path}`')
        self._train_data = self.load_data(self.train_data_path)
        if self.test_data_path is None:
            self._test_data = None
        else:
            self._logger.info(f'Loading test data `{self.test_data_path}`')
            self._test_data = self.load_data(self.test_data_path)

        # Create collate_fn and transforms
        self._initialize_dataset_and_dataloader_deps()

        # Create dataloaders
        self._train_dataloader = self._create_dataloader(self.train_data, True)
        if self.test_data is None:
            self._test_dataloader = None
        else:
            self._test_dataloader = self._create_dataloader(
                self.test_data,
                False
            )

        # Create model
        self._model = self._create_model()

        if print_summary:
            self.model.summary()

        # Get device
        self._device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'  # type: ignore
        )
        self._model = self.model.to(self.device)

        # Create optimizer
        self._optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.hparams['learning_rate']
        )

        # Create loss function
        self._criterion = nn.CrossEntropyLoss(reduction='mean').to(self.device)

    @property
    def train_data_path(self) -> str:
        """
        Return property `train_data_path`.

        :return: path to the training dataset.
        """
        return self._train_data_path

    @property
    def test_data_path(self) -> Optional[str]:
        """
        Return property `test_data_path`.

        :return: path to the development or test dataset in train or
            evaluation mode, respectively.
        """
        return self._test_data_path

    @property
    def models_dir(self) -> Optional[str]:
        """
        Return property `models_dir`.

        :return: path to the directory where models are saved each epoch.
        """
        return self._models_dir

    @property
    def logs_dir(self) -> Optional[str]:
        """
        Return property `logs_dir`.

        :return: path to the directory where logs are saved each epoch.
        """
        return self._logs_dir

    @property
    def checkpoint_path(self) -> Optional[str]:
        """
        Return property `checkpoint_path`.

        :return: path to the model checkpoint to be evaluated or to
            continue in a training from it.
        """
        return self._checkpoint_path

    @property
    def num_workers(self) -> int:
        """
        Return property `num_workers`.

        :return: how many subprocesses to use for data loading.
        """
        return self._num_workers

    @property
    def print_summary(self) -> bool:
        """
        Return property `print_summary`.

        :return: whether to print model summary.
        """
        return self._print_summary

    @property
    def hparams(self) -> Dict[str, Any]:
        """
        Return property `hparams`.

        :return: dictionary with hyperparameters for model to be trained.
        """
        return self._hparams

    @property
    def train_data(self) -> pd.DataFrame:
        """
        Return property `train_data`.

        :return: loaded train data.
        """
        return self._train_data

    @property
    def test_data(self) -> Optional[pd.DataFrame]:
        """
        Return property `test_data`.

        :return: loaded test data.
        """
        return self._test_data

    @property
    def train_dataloader(self) -> DataLoader:
        """
        Return property `train_dataloader`.

        :return: train dataloader.
        """
        return self._train_dataloader

    @property
    def test_dataloader(self) -> Optional[DataLoader]:
        """
        Return property `test_dataloader`.

        :return: test dataloader.
        """
        return self._test_dataloader

    @property
    def model(self) -> nn.Module:
        """
        Return property `model`.

        :return: model to be trained.
        """
        return self._model

    @property
    def device(self) -> torch.device:
        """
        Return property `device`.

        :return: device to be model trained on.
        """
        return self._device

    @property
    def optimizer(self) -> Optimizer:
        """
        Return property `optimizer`.

        :return: optimization algorithm.
        """
        return self._optimizer

    @property
    def criterion(self) -> _Loss:
        """
        Return property `criterion`.

        :return: loss function.
        """
        return self._criterion

    @staticmethod
    def load_data(data_path: str) -> pd.DataFrame:
        """
        Load data from `data_path`.

        :param data_path: path to the `.csv` data file.
        :return: loaded data.
        """
        data = pd.read_csv(data_path)
        assert isinstance(data, pd.DataFrame)
        return data

    def _initialize_dataset_and_dataloader_deps(self) -> None:
        """
        Initialize dataset and dataloader dependencies.

        Method set `self._transforms` and `self._collate_fn`.
        """
        if self._model_class != Transformer:
            # Get tokenizer and build vocabulary from train data
            self._logger.info('Building vocabulary')
            tokenizer = get_tokenizer(
                self.hparams['tokenizer'],
                language=self.hparams['tokenizer_language']
            )
            tok_list = [
                tokenizer(sentence) for sentence in
                np.concatenate(
                    [
                        self.train_data.premise.values,
                        self.train_data.hypothesis.values
                    ]
                )
            ]

            self.vocab = build_vocab_from_iterator(tok_list)
            self._logger.info('Loading vectors')
            self.vocab.load_vectors(self.hparams['vectors'])

            # Define transformations
            self._transforms = Compose([
                Tokenize(tokenizer, 'premise', 'hypothesis'),
                ToVocabulary(self.vocab, 'premise', 'hypothesis'),
                ToTensor(torch.long, 'premise', 'hypothesis', 'label')
            ])

            # Define collate_fn
            self._collate_fn = batch_fn
        else:
            # Define transformations
            self._transforms = Compose([ToTensor(torch.long, 'label')])

            # Get tokenizer
            transformer_tokenizer = AutoTokenizer.from_pretrained(
                self.hparams['pretrained_transformer']
            )

            # Define collate_fn
            self._collate_fn = partial(
                batch_fn,
                transformer_tokenizer=transformer_tokenizer
            )

    def _create_dataloader(
        self,
        data: pd.DataFrame,
        shuffle: bool
    ) -> DataLoader:
        """
        Create dataloader for given `data`.

        :param data: data to create dataloader from.
        :param shuffle: whether to shuffle data in dataloader or not.
        :return: dataloader for given `data`.
        """
        dataset = NLIDataset(data, self._transforms)

        return DataLoader(
            dataset,
            batch_size=self.hparams['batch_size'],
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )

    def _create_model(self) -> nn.Module:
        """
        Create model.

        :return: created model.
        :raise:
            NotImplementedError: when branch for given model is not
                implemented.
        """
        if self._model_class == BiLSTMModel:
            model = self._model_class(
                embeddings=self.vocab.vectors,  # type: ignore
                lstm_hidden_size=self.hparams['lstm_hidden_size'],
                max_pooling=self.hparams['max_pooling'],
                dropout_prob=self.hparams['dropout_prob']
            )
        elif self._model_class == Transformer:
            model = self._model_class(
                pretrained_model_name_or_path=self.hparams[
                    'pretrained_transformer'
                ],
                dropout_prob=self.hparams['dropout_prob']
            )
        else:
            raise NotImplementedError(
                f'Branch for model `{self._model_class}` is not implemented.'
            )
        return model

    def train(self) -> None:
        """Train a new model."""
        self._logger.info('Train eval loop')
        assert isinstance(self.train_dataloader, DataLoader)

        epochs = self.hparams['epochs']
        best_train_metrics, best_test_metrics = {}, {}
        best_train_acc, best_test_acc = -1., -1.
        writer = None if self.logs_dir is None else \
            SummaryWriter(self.logs_dir)

        for epoch in range(1, epochs + 1):
            train_metrics = train_fn(
                self.model,
                self.device,
                self.train_dataloader,
                self.criterion,
                self.optimizer,
                epoch,
                epochs,
                writer
            )
            if train_metrics['accuracy'] > best_train_acc:
                best_train_acc = train_metrics['accuracy']
                best_train_metrics = train_metrics
            if self.test_dataloader is not None:
                test_metrics = test_fn(
                    self.model,
                    self.device,
                    self.test_dataloader,
                    self.criterion,
                    epoch,
                    writer
                )
                if test_metrics['accuracy'] > best_test_acc:
                    best_test_acc = test_metrics['accuracy']
                    best_test_metrics = test_metrics
            if self.models_dir is not None:
                os.makedirs(self.models_dir, exist_ok=True)
                model_path = os.path.join(
                    self.models_dir,
                    f'model_{epoch:03}.pt'
                )
                self._logger.info(f'Saving model `{model_path}`')
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()
                    },
                    model_path
                )

        # Write hyperparameters
        if writer is not None:
            best_train_metrics = {
                'train_' + key: value for key, value in
                best_train_metrics.items()
            }
            best_test_metrics = {
                'dev_' + key: value for key, value in
                best_test_metrics.items()
            }
            writer.add_hparams(
                self.hparams,
                {**best_train_metrics, **best_test_metrics}
            )

    def evaluate(self) -> None:
        """Evaluate a model from `self.checkpoint_path`."""
        self._logger.info('Evaluation')
        assert isinstance(self.test_dataloader, DataLoader)
        assert isinstance(self.checkpoint_path, str)

        writer = None if self.logs_dir is None else \
            SummaryWriter(os.path.join(self.logs_dir, 'eval'))

        # Load checkpoint
        self._logger.info(f'Loading checkpoint {self.checkpoint_path}')
        checkpoint = torch.load(self.checkpoint_path)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']

        test_metrics = test_fn(
            self.model,
            self.device,
            self.test_dataloader,
            self.criterion,
            epoch,
            writer
        )

        # Write hyperparameters
        if writer is not None:
            test_metrics = {
                'test_' + key: value for key, value in
                test_metrics.items()
            }
            writer.add_hparams(
                self.hparams,
                test_metrics
            )
