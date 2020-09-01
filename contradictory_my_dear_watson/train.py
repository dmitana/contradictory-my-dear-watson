import time

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_fn(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    criterion: _Loss,
    optimizer: Optimizer,
    epoch: int,
    epochs: int
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
    """
    # Start timer
    start_time = time.time()

    # Set train mode
    model.train()

    # Create progress bar
    loop = tqdm(train_loader)
    for data in loop:
        # Send data to device
        premise = data['premise'].to(device)
        hypothesis = data['hypothesis'].to(device)
        label = data['label'].to(device)

        # Reset all gradients
        optimizer.zero_grad()

        # Run forward pass
        output = model(premise, hypothesis)

        # Compute loss
        loss = criterion(output, label)

        # Run backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Show progress
        elapsed_time = time.time() - start_time
        loop.set_description(f'Epoch {epoch}/{epochs}')
        loop.set_postfix(loss=loss.item(), elapsed_time=f'{elapsed_time:.2f}s')