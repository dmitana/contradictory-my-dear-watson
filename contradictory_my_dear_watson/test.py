import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from contradictory_my_dear_watson.metrics import Accuracy, AvgLoss


def test_fn(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    criterion: _Loss
):
    """
    Evaluate a given `model`.

    :param model: model to be trained.
    :param device: device to be model trained on.
    :param test_loader: data loader with test data.
    :param criterion: loss function.
    """
    # Set eval mode
    model.eval()

    acc = Accuracy()
    avg_loss = AvgLoss()

    with torch.no_grad():
        for data in test_loader:
            # Send data to device
            premise = data['premise'].to(device)
            hypothesis = data['hypothesis'].to(device)
            label = data['label'].to(device)

            # Run forward pass
            output = model(premise, hypothesis)

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
