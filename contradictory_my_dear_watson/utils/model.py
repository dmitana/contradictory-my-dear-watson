from typing import Callable

from torch import Tensor
from torch.nn.utils.rnn import PackedSequence


def packed_sequence_element_wise_apply(
    fn: Callable[[Tensor], Tensor],
    packed_sequence: PackedSequence
) -> PackedSequence:
    """
    Apply a pointwise function `fn` to each element in `packed_sequence`.

    Source: https://discuss.pytorch.org/t/how-to-use-pack-sequence-if-we-
    are-going-to-use-word-embedding-and-bilstm/28184

    :param fn: function to be applied to each element in
        `packed_sequence`.
    :param packed_sequence: packed sequence to be modified.
    :return: modified packed sequence.
    """
    return PackedSequence(
        fn(packed_sequence.data),
        packed_sequence.batch_sizes
    )
