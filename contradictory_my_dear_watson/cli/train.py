from argparse import Namespace
import logging

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import DataLoader
from torchsummaryX import summary
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchvision.transforms import Compose

from contradictory_my_dear_watson.datasets import NLIDataset
from contradictory_my_dear_watson.models import BiLSTMModel
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
    # Load train data
    logger.info(f'Loading train data `{args.train_data_path}`')
    train_data = pd.read_csv(args.train_data_path)
    assert isinstance(train_data, pd.DataFrame)

    # Get tokenizer and build vocabulary from train data
    logger.info('Building vocabulary')
    tokenizer = get_tokenizer(args.tokenizer, language=args.tokenizer_language)
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

    # Create train dataset
    train_dataset = NLIDataset(train_data, vocab, transforms)

    # Create train data loader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=batch_fn
    )

    # Load dev data
    if args.dev_data_path is not None:
        logger.info(f'Loading dev data `{args.dev_data_path}`')
        dev_data = pd.read_csv(args.dev_data_path)
        assert isinstance(dev_data, pd.DataFrame)

        # Create dev dataset
        dev_dataset = NLIDataset(dev_data, vocab, transforms)

        # Create dev data loader
        dev_dataloader = DataLoader(
            dev_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=batch_fn
        )
    else:
        dev_dataloader = None

    # Create model
    if args.model == BiLSTMModel:
        model = args.model(
            embeddings=vocab.vectors,
            lstm_hidden_size=args.lstm_hidden_size,
            dropout_prob=args.dropout_prob
        )
    else:
        raise NotImplementedError(
            f'Branch for model `{args.model}` is not implemented.'
        )

    if args.print_summary:
        summary(
            model,
            pack_sequence(
                torch.zeros(1, 20, dtype=torch.long),  # type: ignore
                enforce_sorted=False
            ),
            pack_sequence(
                torch.zeros(1, 20, dtype=torch.long),  # type: ignore
                enforce_sorted=False
            ),
        )

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
    for epoch in range(1, epochs + 1):
        train_fn(
            model,
            device,
            train_dataloader,
            criterion,
            optimizer,
            epoch,
            epochs
        )
        if dev_dataloader is not None:
            test_fn(model, device, dev_dataloader, criterion)
