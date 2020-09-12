from typing import Any, Dict, List, Optional

import torch
from torch.nn.utils.rnn import pack_sequence
from transformers import PreTrainedTokenizer


def batch_fn(
    samples: List[Dict[str, Any]],
    transformer_tokenizer: Optional[PreTrainedTokenizer] = None,
    premise_key: str = 'premise',
    hypothesis_key: str = 'hypothesis',
    label_key: str = 'label'
) -> Dict[str, Any]:
    """
    Merge a list of `samples` to form a mini-batch of Tensors.

    :param samples: list of dataset samples of size `batch_size`.
    :param transformer_tokenizer: tokenizer for specific `Transformer`
        model.
    :param premise_key: key for premise.
    :param hypothesis_key: key for hypothesis.
    :param label_key: key for label.
    :return: dict containing mini-batch.
    """
    # Create lists of items
    premises = [sample[premise_key] for sample in samples]
    hypotheses = [sample[hypothesis_key] for sample in samples]
    labels = [sample[label_key] for sample in samples]

    if transformer_tokenizer is not None:
        result = {
            'inputs': transformer_tokenizer(
                premises,
                hypotheses,
                padding=True,
                return_tensors='pt'
            ),
            label_key: torch.tensor(labels)  # type: ignore
        }
    else:
        # Create packed sequences for variable length texts and tensor
        result = {
            'inputs': {
                premise_key: pack_sequence(premises, enforce_sorted=False),
                hypothesis_key: pack_sequence(
                    hypotheses,
                    enforce_sorted=False
                ),
            },
            label_key: torch.tensor(labels)  # type: ignore
        }

    return result
