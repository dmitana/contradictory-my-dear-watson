import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader


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
    # Define default values
    test_loss, correct, n_batchs = 0, 0, 0

    # Set eval mode
    model.eval()

    with torch.no_grad():
        for data in test_loader:
            n_batchs += 1

            # Send data to device
            premise = data['premise'].to(device)
            hypothesis = data['hypothesis'].to(device)
            label = data['label'].to(device)

            # Run forward pass
            output = model(premise, hypothesis)

            # Sum up batch loss
            test_loss += criterion(output, label).item()

            # Get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

    # Compute average loss
    test_loss /= n_batchs

    # Compute accuracy
    acc = 100. * correct / len(test_loader.dataset)

    print(f'Test set - avg loss: {test_loss:.4f} - accuracy: {acc:.2f}')
