from argparse import Namespace
from functools import partial
import logging
import os

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
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
    ToTensor, ToVocabulary, Tokenize
)
from contradictory_my_dear_watson.utils.dataloader import batch_fn

logger = logging.getLogger(__name__)


def train(args: Namespace) -> None:
    """
    Train a new model.

    :param args: train command's arguments.
    :raise:
        NotImplementedError: when branch for given model is not
            implemented.
    """
    # Define basic training hparams
    hparams = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate
    }

    # Load train data
    logger.info(f'Loading train data `{args.train_data_path}`')
    train_data = pd.read_csv(args.train_data_path)
    assert isinstance(train_data, pd.DataFrame)

    if args.model != Transformer:
        # Get tokenizer and build vocabulary from train data
        logger.info('Building vocabulary')
        tokenizer = get_tokenizer(
            args.tokenizer,
            language=args.tokenizer_language
        )
        tok_list = [
            tokenizer(sentence) for sentence in
            np.concatenate(
                [train_data.premise.values, train_data.hypothesis.values]
            )
        ]

        vocab = build_vocab_from_iterator(tok_list)
        logger.info('Loading vectors')
        vocab.load_vectors(args.vectors)

        # Define transformations
        transforms = Compose([
            Tokenize(tokenizer, 'premise', 'hypothesis'),
            ToVocabulary(vocab, 'premise', 'hypothesis'),
            ToTensor(torch.long, 'premise', 'hypothesis', 'label')
        ])

        # Define collate_fn
        collate_fn = batch_fn
    else:
        # Define transformations
        transforms = Compose([ToTensor(torch.long, 'label')])

        # Get tokenizer
        transformer_tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_transformer
        )

        # Define collate_fn
        collate_fn = partial(
            batch_fn,
            transformer_tokenizer=transformer_tokenizer
        )

    # Create train dataset
    train_dataset = NLIDataset(train_data, transforms)

    # Create train data loader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    # Load dev data
    if args.dev_data_path is not None:
        logger.info(f'Loading dev data `{args.dev_data_path}`')
        dev_data = pd.read_csv(args.dev_data_path)
        assert isinstance(dev_data, pd.DataFrame)

        # Create dev dataset
        dev_dataset = NLIDataset(dev_data, transforms)

        # Create dev data loader
        dev_dataloader = DataLoader(
            dev_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )
    else:
        dev_dataloader = None

    # Create model
    if args.model == BiLSTMModel:
        model = args.model(
            embeddings=vocab.vectors,  # type: ignore
            lstm_hidden_size=args.lstm_hidden_size,
            max_pooling=args.max_pooling,
            dropout_prob=args.dropout_prob
        )
        hparams.update({
            'vectors': args.vectors,
            'tokenizer': args.tokenizer,
            'tokenizer_language': args.tokenizer_language
        })
    elif args.model == Transformer:
        model = args.model(
            pretrained_model_name_or_path=args.pretrained_transformer,
            dropout_prob=args.dropout_prob
        )
    else:
        raise NotImplementedError(
            f'Branch for model `{args.model}` is not implemented.'
        )

    if args.print_summary:
        model.summary()

    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu'  # type: ignore
    )
    model = model.to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Create loss function
    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)

    logger.info('Train eval loop')
    epochs = args.epochs
    best_train_metrics, best_test_metrics = {}, {}
    best_train_acc, best_test_acc = -1., -1.
    writer = None if args.logs_dir is None else SummaryWriter(args.logs_dir)
    for epoch in range(1, epochs + 1):
        train_metrics = train_fn(
            model,
            device,
            train_dataloader,
            criterion,
            optimizer,
            epoch,
            epochs,
            writer
        )
        if train_metrics['accuracy'] > best_train_acc:
            best_train_acc = train_metrics['accuracy']
            best_train_metrics = train_metrics
        if dev_dataloader is not None:
            test_metrics = test_fn(
                model,
                device,
                dev_dataloader,
                criterion,
                epoch,
                writer
            )
            if test_metrics['accuracy'] > best_test_acc:
                best_test_acc = test_metrics['accuracy']
                best_test_metrics = test_metrics
        if args.models_dir is not None:
            os.makedirs(args.models_dir, exist_ok=True)
            model_path = os.path.join(args.models_dir, f'model_{epoch:03}.pt')
            logger.info(f'Saving model `{model_path}`')
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
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
            'test_' + key: value for key, value in
            best_test_metrics.items()
        }
        writer.add_hparams(
            {**hparams, **model.hparams},
            {**best_train_metrics, **best_test_metrics}
        )
