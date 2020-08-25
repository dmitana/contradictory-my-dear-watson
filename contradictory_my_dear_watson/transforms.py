from typing import Any, Callable, Dict, Sequence

import torch
from torch import Tensor
from torchtext.vocab import Vocab


class Tokenize():
    """
    Tokenize transformation.

    Tokenize sentences.

    :param self.tokenizer: Callable, tokenizer which performs
        tokenization.
    :param self.fields: sequence of str, fields in a `sample` to be
        tokenized.
    """

    def __init__(self, tokenizer: Callable, *fields: str):
        """
        Create a new instance of `Tokenize`.

        :param tokenizer: tokenizer which performs tokenization.
        :param *fields: fields in a `sample` to be tokenized.
        """
        self._tokenizer = tokenizer
        self._fields = fields

    @property
    def tokenizer(self) -> Callable:
        """
        Return property `tokenizer`.

        :return: tokenizer which performs tokenization.
        """
        return self._tokenizer

    @property
    def fields(self) -> Sequence[str]:
        """
        Return property `fields`.

        :return: fields in a `sample` to be tokenized.
        """
        return self._fields

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenize `self.fields` in a `sample` using `self.tokenizer`.

        :param sample: dataset sample.
        :return: tokenized `sample`.
        """
        other_fields = set(sample.keys()).difference(self.fields)

        result = {
            field: self.tokenizer(sample[field]) for field in self.fields
        }

        for field in other_fields:
            result[field] = sample[field]

        return result


class ToVocabulary():
    """
    To vocabulary transformation.

    Convert tokens to vocabulary indices.

    :param self.vocabulary: torchtext.vocab.Vocab, vocabulary created
        from training data.
    :param self.fields: sequence of str, fields in a `sample` to be
        converted to vocabulary indices.
    """

    def __init__(self, vocabulary: Vocab, *fields: str):
        """
        Crate a new instance of `ToVocabulary`.

        :param vocabulary: vocabulary created from training data.
        :param *fields: sequence of str, fields in a `sample` to be
            converted to vocabulary indices.
        """
        self._vocabulary = vocabulary
        self._fields = fields

    @property
    def vocabulary(self) -> Vocab:
        """
        Return property `vocabulary`.

        :return: vocabulary created from training data.
        """
        return self._vocabulary

    @property
    def fields(self) -> Sequence[str]:
        """
        Return property `fields`.

        :return: fields in a `sample` to be tokenized.
        """
        return self._fields

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert `self.fields` in a `sample` using `self.vocabulary`.

        Convert tokens in `self.fields` from a `sample` to vocabulary
        indices using `self.vocabulary`.

        :param sample: dataset sample.
        :return: converted `sample`.
        """
        other_fields = set(sample.keys()).difference(self.fields)

        result = {
            field: [
                self.vocabulary[token] for token in sample[field]
            ] for field in self.fields
        }

        for field in other_fields:
            result[field] = sample[field]

        return result


class ToTensor():
    """
    To tensor transformation.

    Convert data to tensors.

    :param self.dtype: pytorch data type.
    :param self.fields: sequence of str, fields in a `sample` to be
        tokenized.
    """

    def __init__(self, dtype, *fields: str):
        """
        Create a new instance of `ToTensor`.

        :param dtype: pytorch data type.
        :param *fields: fields in a `sample` to be tokenized.
        """
        self._dtype = dtype
        self._fields = fields

    @property
    def dtype(self):
        """
        Return property `dtype`.

        :return: pytorch data type.
        """
        return self._dtype

    @property
    def fields(self) -> Sequence[str]:
        """
        Return property `fields`.

        :return: fields in a `sample` to be tokenized.
        """
        return self._fields

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Tensor]:
        """
        Convert `self.fields` in `sample` to tensor of type `self.dtype`.

        :param sample: dataset sample.
        :return: converted `sample`.
        """
        return {
            field: torch.tensor(
                sample[field]
            ).to(dtype=self.dtype) for field in self.fields  # type: ignore
        }
