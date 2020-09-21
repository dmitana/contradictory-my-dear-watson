from typing import Optional

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from contradictory_my_dear_watson.metrics import Accuracy, AvgLoss
from contradictory_my_dear_watson.utils import logging


def test_fn(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    criterion: _Loss,
    epoch: int,
    writer: Optional[SummaryWriter] = None
):
    """
    Evaluate a given `model`.

    :param model: model to be trained.
    :param device: device to be model trained on.
    :param test_loader: data loader with test data.
    :param criterion: loss function.
    :param epoch: current epoch.
    :param writer: TensorBoard writer.
    """
    # Set eval mode
    model.eval()

    acc = Accuracy()
    avg_loss = AvgLoss()

    with torch.no_grad():
        for data in test_loader:
            # Send data to device
            label = data['label'].to(device)
            inputs = data['inputs']
            if isinstance(inputs, dict):
                inputs['premise'] = inputs['premise'].to(device)
                inputs['hypothesis'] = inputs['hypothesis'].to(device)
            else:
                inputs = inputs.to(device)

            # Run forward pass
            output = model(inputs)

            # Compute loss
            loss = criterion(output, label)

            # Update accuracy and loss
            acc.update(output, label)
            avg_loss.update(loss.item())

    # Compute average accuracy and loss
    acc_val = 100. * acc.compute().result
    avg_loss_val = avg_loss.compute().result

    print(
        f'Test set - avg loss: {avg_loss_val:.4f} - accuracy: {acc_val:.2f} %'
    )

    # Log evaluation
    if writer is not None:
        scalars = {'loss': avg_loss_val, 'accuracy': acc_val}
        logging.tensorboard_add_scalars(writer, 'test', scalars, epoch)
