import argparse

from contradictory_my_dear_watson.cli.evaluate import evaluate
from contradictory_my_dear_watson.cli.train import train


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create argument parser with subparsers for commands.

    :return: argument parser.
    """
    parser = argparse.ArgumentParser(
        prog='contradictory-my-dear-watson',
        description='Program to solve the Contradictory, My Dear Watson '
                    'problem.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Subparsers
    subparsers = parser.add_subparsers(
        title='commands',
        description='use commands to perform various actions',
        help='additional help for commands'
    )
    create_train_subparser(subparsers)
    create_evaluate_subparser(subparsers)

    return parser


def create_train_subparser(subparsers: argparse._SubParsersAction) -> None:
    """
    Create parser for the `train` command.

    :param subparsers: argparse subparsers where a new parser will be
        added.
    """
    parser_train = subparsers.add_parser(
        'train',
        help='train a new model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required named arguments
    parser_required_named = parser_train.add_argument_group(
        'required named arguments'
    )
    parser_required_named.add_argument(
        '--train-data-path',
        required=True,
        type=str,
        help='Path to the training dataset.'

    )

    # Hyperparameters agruments
    parser_hparams = parser_train.add_argument_group(
        'hyperparameters'
    )
    parser_hparams.add_argument(
        '--lstm-hidden-size',
        type=int,
        default=256,
        help='Number of neurons in a LSTM layer.'
    )
    parser_hparams.add_argument(
        '--learning-rate',
        type=float,
        default=0.01,
        help='Learning rate.'
    )
    parser_hparams.add_argument(
        '--tokenizer',
        type=str,
        default='spacy',
        help='Tokenizer name to be used by PyTorch `get_tokenizer`'
             'function to get instance of `Tokenizer`.'
    )
    parser_hparams.add_argument(
        '--tokenizer-language',
        type=str,
        default='en_core_web_sm',
        help='Tokenizer language to be used by PyTorch `get_tokenizer`'
             'function to get instance of `Tokenizer`.'
    )
    # TODO: add choices
    parser_hparams.add_argument(
        '--vectors',
        type=str,
        default='fasttext.simple.300d',
        help='Vectors name to be loaded by PyTorch `load_vectors` method.'
    )
    parser_hparams.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Size of one mini-batch.'
    )
    parser_hparams.add_argument(
        '--epochs',
        type=int,
        default=1,
        help='Number of training epochs.'
    )
    parser_hparams.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='How many subprocesses to use for data loading. `0` means '
             'that the data will be loaded in the main process.'
    )
    parser_hparams.add_argument(
        '--print-summary',
        action='store_true',
        help='Whether to print model summary.'
    )

    # Add execution function
    parser_train.set_defaults(func=train)


def create_evaluate_subparser(subparsers: argparse._SubParsersAction) -> None:
    """
    Create parser for the `evaluate` command.

    :param subparsers: argparse subparsers where a new parser will be
        added.
    """
    parser_evaluate = subparsers.add_parser(
        'evaluate',
        help='evaluate a given model'
    )
    parser_evaluate.set_defaults(func=evaluate)
