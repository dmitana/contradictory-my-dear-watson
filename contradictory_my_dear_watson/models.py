from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import PackedSequence

from contradictory_my_dear_watson.utils.model import \
    packed_sequence_element_wise_apply


class BiLSTMModel(nn.Module):
    """
    Base Bi-LSTM model.

    To encode `premise` and `hypothesis` it concatenates forward and
    backward directions of Bi-LSTM.

    :param self.embedding: torhc.nn.Embedding, pre-trained embedding
        layer.
    :param self.lstm: torch.nn.LSTM, Bi-LSTM layer.
    :param self.fc: torch.nn.Linear, final classification layer.
    """

    def __init__(
        self,
        num_embbedings: int = 0,
        embedding_dim: int = 300,
        embeddings: Optional[Tensor] = None,
        freeze_embeddings: bool = True,
        lstm_hidden_size: int = 128,
        dropout_prob: float = 0.5,
        num_classes: int = 3,
    ):
        """
        Create a new instance of `BiLSTMModel`.

        :param num_embbedings: size of the dictionary of embeddings.
        :param embedding_dim: the size of each embedding vector.
        :param embeddings: FloatTensor containing weights for the
            Embedding. First dimension is being passed to Embedding as
            `num_embeddings`, second as `embedding_dim`.
        :param freeze_embeddings: If `True`, the tensor does not get
            updated in the learning process. Equivalent to
            `embedding.weight.requires_grad = False`.
        :param lstm_hidden_size: the number of features in the hidden
            state `h`.
        :param dropout_prob: probability of an element to be zeroed.
        :param num_classes: number of classes to be predicted.
        """
        super(BiLSTMModel, self).__init__()
        if embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embeddings=embeddings,
                freeze=freeze_embeddings
            )
        else:
            self.embedding = nn.Embedding(
                num_embeddings=num_embbedings,
                embedding_dim=embedding_dim
            )
        self.lstm = nn.LSTM(
            input_size=embeddings.size(1),
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(8 * lstm_hidden_size, num_classes)

    def forward(
        self,
        premise: PackedSequence,
        hypothesis: PackedSequence
    ) -> Tensor:
        """
        Execute forward pass of the `BiLSTMModel`.

        :param premise: input premise.
        :param hypothesis: input hypothesis
        :return: final layer output. Classification without Softmax
            applied.
        """
        # Get embeddings
        premise_emb = packed_sequence_element_wise_apply(
            self.embedding,
            premise
        )
        hypothesis_emb = packed_sequence_element_wise_apply(
            self.embedding,
            hypothesis
        )

        # Encode premise and hypothesis, use only the last hidden state
        _, (premise_hn, _) = self.lstm(premise_emb)
        _, (hypothesis_hn, _) = self.lstm(hypothesis_emb)

        # Concat forward and backward directions
        premise_hn = premise_hn.transpose(1, 0).contiguous()
        premise_hn = premise_hn.view(len(premise_hn), -1)
        hypothesis_hn = hypothesis_hn.transpose(1, 0).contiguous()
        hypothesis_hn = hypothesis_hn.view(len(hypothesis_hn), -1)

        # Restore ordering changed by sequence packing
        premise_hn = premise_hn[premise.unsorted_indices]
        hypothesis_hn = hypothesis_hn[hypothesis.unsorted_indices]

        # Compute element-wise absolute difference and product between
        # premise and hypothesis
        ph_abs_diff = torch.abs(premise_hn - hypothesis_hn)  # type: ignore
        ph_mul = torch.mul(premise_hn, hypothesis_hn)  # type: ignore

        # Create final feature vector
        features = torch.cat(  # type: ignore
            (premise_hn, hypothesis_hn, ph_abs_diff, ph_mul),
            1
        )

        # Add dropout
        features = self.dropout(features)

        # Classify using FC layer
        out = self.fc(features)

        return out
