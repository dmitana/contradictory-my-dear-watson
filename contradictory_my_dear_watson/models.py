from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

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
    :param self.max_pooling: bool, whether max-pooling over LSTM's output
        is enabled or not. If max-pooling is not enabled, only last
        LSTM's hidden states are used.
    :param self.dropout: torch.nn.Dropout, dropout.
    :param self.act: linear layers' activation.
    :param self.fc: torch.nn.Linear, fully-connected layer.
    :param self.fc_out: torch.nn.Linear, final classification layer.
    """

    def __init__(
        self,
        num_embbedings: int = 0,
        embedding_dim: int = 300,
        embeddings: Optional[Tensor] = None,
        freeze_embeddings: bool = True,
        lstm_hidden_size: int = 128,
        max_pooling: bool = False,
        dropout_prob: float = 0.5,
        num_classes: int = 3
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
        :param max_pooling: whether max-pooling over LSTM's output is
            enabled or not. If max-pooling is not enabled, only last
            LSTM's hidden states are used.
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
        self.max_pooling = max_pooling

        self.dropout = nn.Dropout(p=dropout_prob)
        self.act = nn.ReLU()
        self.fc = nn.Linear(8 * lstm_hidden_size, 512)
        self.fc_out = nn.Linear(512, num_classes)

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
        premise_out, (premise_hn, _) = self.lstm(premise_emb)
        hypothesis_out, (hypothesis_hn, _) = self.lstm(hypothesis_emb)

        if self.max_pooling:
            # Pad with big negative value because of max-pooling
            premise_out, _ = pad_packed_sequence(
                premise_out,
                batch_first=True,
                padding_value=-1e9
            )
            premise_out, _ = torch.max(premise_out, 1)  # type: ignore

            hypothesis_out, _ = pad_packed_sequence(
                hypothesis_out,
                batch_first=True,
                padding_value=-1e9
            )
            hypothesis_out, _ = torch.max(hypothesis_out, 1)  # type: ignore

            # Restore ordering changed by sequence packing
            premise_out = premise_out[premise.unsorted_indices]
            hypothesis_out = hypothesis_out[hypothesis.unsorted_indices]

            premise_enc = premise_out
            hypothesis_enc = hypothesis_out
        else:
            # Concat forward and backward directions
            premise_hn = premise_hn.transpose(1, 0).contiguous()
            premise_hn = premise_hn.view(len(premise_hn), -1)

            hypothesis_hn = hypothesis_hn.transpose(1, 0).contiguous()
            hypothesis_hn = hypothesis_hn.view(len(hypothesis_hn), -1)

            # Restore ordering changed by sequence packing
            premise_hn = premise_hn[premise.unsorted_indices]
            hypothesis_hn = hypothesis_hn[hypothesis.unsorted_indices]

            premise_enc = premise_hn
            hypothesis_enc = hypothesis_hn

        # Compute element-wise absolute difference and product between
        # premise and hypothesis
        ph_abs_diff = torch.abs(premise_enc - hypothesis_enc)  # type: ignore
        ph_mul = torch.mul(premise_enc, hypothesis_enc)  # type: ignore

        # Create final feature vector
        features = torch.cat(  # type: ignore
            (premise_enc, hypothesis_enc, ph_abs_diff, ph_mul),
            1
        )

        # Classify using FC layers
        features = self.act(self.fc(features))
        features = self.dropout(features)
        out = self.fc_out(features)

        return out
