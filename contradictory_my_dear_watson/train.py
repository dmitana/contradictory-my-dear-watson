import time
from typing import Optional

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from contradictory_my_dear_watson.metrics import Accuracy, AvgLoss
from contradictory_my_dear_watson.utils import logging


def train_fn(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    criterion: _Loss,
    optimizer: Optimizer,
    epoch: int,
    epochs: int,
    writer: Optional[SummaryWriter] = None
) -> None:
    """
    Train a given `model`.

    :param model: model to be trained.
    :param device: device to be model trained on.
    :param train_loader: data loader with train data.
    :param criterion: loss function.
    :param optimizer: optimization algorithm.
    :param epoch: current epoch.
    :param epochs: total number of epochs.
    :param writer: TensorBoard writer.
    """
    # Start timer
    start_time = time.time()

    # Set train mode
    model.train()

    acc = Accuracy()
    avg_loss = AvgLoss()

    # Create progress bar
    loop = tqdm(train_loader)
    for data in loop:
        # Send data to device
        label = data['label'].to(device)
        inputs = data['inputs']
        if isinstance(inputs, dict):
            inputs['premise'] = inputs['premise'].to(device)
            inputs['hypothesis'] = inputs['hypothesis'].to(device)
        else:
            inputs = inputs.to(device)

        # Reset all gradients
        optimizer.zero_grad()

        # Run forward pass
        output = model(inputs)

        # Compute loss
        loss = criterion(output, label)

        # Run backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Update accuracy/loss and compute avg accuracy/loss
        acc.update(output, label).compute()
        avg_loss.update(loss.item()).compute()

        # Show progress
        elapsed_time = time.time() - start_time
        loop.set_description(f'Epoch {epoch}/{epochs}')
        loop.set_postfix(
            avg_loss=avg_loss.result,
            avg_acc=100. * acc.result,
            elapsed_time=f'{elapsed_time:.2f}s'
        )

    # Log progress
    if writer is not None:
        scalars = {'loss': avg_loss.result, 'accuracy': 100. * acc.result}
        logging.tensorboard_add_scalars(writer, 'train', scalars, epoch)
