import argparse
import logging
from typing import Any, Callable, Dict, List, Optional

from contradictory_my_dear_watson.cli.evaluate import evaluate
from contradictory_my_dear_watson.cli.train import train
from contradictory_my_dear_watson.models import BiLSTMModel


def arg_map(**mapping: Dict[Any, Any]) -> Callable[[Any], Any]:
    """
    Create function to map arguments using given `mapping`.

    :param mapping: mapping between input arguments and their desired
        values.
    :return: function to perform argument mapping.
    """
    def parse_argument(arg: Any) -> Any:
        if arg in mapping:
            return mapping[arg]
        else:
            msg = "invalid choice: {!r} (choose from {})"
            choices = ", ".join(
                sorted(repr(choice) for choice in mapping.keys())
            )
            raise argparse.ArgumentTypeError(msg.format(arg, choices))

    return parse_argument


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create argument parser with subparsers for commands.

    :return: argument parser.
    """
    # Create main parser
    parser = argparse.ArgumentParser(
        prog='contradictory-my-dear-watson',
        description='Program to solve the Contradictory, My Dear Watson '
                    'problem.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Create parent parser for shared arguments
    parent_parser = argparse.ArgumentParser(
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Optional arguments
    parent_parser.add_argument(
        '--logging-level',
        type=arg_map(**logging_level_mapping),
        metavar=f'{{{",".join(key for key in logging_level_mapping.keys())}}}',
        default='info',
        help='Logging level.'
    )

    # Subparsers
    subparsers = parser.add_subparsers(
        title='commands',
        required=True,
        description='use commands to perform various actions',
        help='additional help for commands'
    )
    create_train_subparser(subparsers, [parent_parser])
    create_evaluate_subparser(subparsers, [parent_parser])

    return parser


def create_train_subparser(
    subparsers: argparse._SubParsersAction,
    parents: Optional[List[argparse.ArgumentParser]] = None
) -> None:
    """
    Create parser for the `train` command.

    :param subparsers: argparse subparsers where a new parser will be
        added.
    :param parents: subparser parents.
    """
    parser_train = subparsers.add_parser(
        'train',
        parents=[] if parents is None else parents,
        help='train a new model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Optional arguments
    parser_train.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='How many subprocesses to use for data loading. `0` means '
             'that the data will be loaded in the main process.'
    )
    parser_train.add_argument(
        '--print-summary',
        action='store_true',
        help='Whether to print model summary.'
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

    # Hyperparameters
    parser_hparams = parser_train.add_argument_group(
        'hyperparameters'
    )
    parser_hparams.add_argument(
        '--model',
        type=arg_map(**model_mapping),
        metavar=f'{{{",".join(key for key in model_mapping.keys())}}}',
        default='baseline',
        help='Model to be trained.'
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
    parser_hparams.add_argument(
        '--vectors',
        type=str,
        default='fasttext.simple.300d',
        choices=['fasttext.simple.300d', 'fasttext.en.300d'],
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

    # Add execution function
    parser_train.set_defaults(func=train)


def create_evaluate_subparser(
    subparsers: argparse._SubParsersAction,
    parents: Optional[List[argparse.ArgumentParser]] = None
) -> None:
    """
    Create parser for the `evaluate` command.

    :param subparsers: argparse subparsers where a new parser will be
        added.
    :param parents: subparser parents.
    """
    parser_evaluate = subparsers.add_parser(
        'evaluate',
        parents=[] if parents is None else parents,
        help='evaluate a given model'
    )
    parser_evaluate.set_defaults(func=evaluate)


# Define mappings
logging_level_mapping = {
    'critical': logging.CRITICAL,
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}

model_mapping = {
    'baseline': BiLSTMModel
}
