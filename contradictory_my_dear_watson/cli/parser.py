import argparse
import logging
from typing import Any, Callable, Dict, List, Optional

from contradictory_my_dear_watson.cli.dataset import dataset
from contradictory_my_dear_watson.cli.evaluate import evaluate
from contradictory_my_dear_watson.cli.train import train
from contradictory_my_dear_watson.datasets import (
    ContradictoryMyDearWatsonDataset,
    ContradictoryMyDearWatsonMultilingualDataset,
    SNLI
)
from contradictory_my_dear_watson.models import BiLSTMModel, Transformer


class MyArgumentParser(argparse.ArgumentParser):
    """Class representing custom argument parser."""

    def convert_arg_line_to_args(self, arg_line: str) -> List[str]:
        """
        Convert argument line to arguments.

        :param arg_line: argument line.
        :return: arguments.
        """
        return arg_line.split()


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
    parser = MyArgumentParser(
        prog='contradictory-my-dear-watson',
        description='Program to solve the Contradictory, My Dear Watson '
                    'problem.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@'
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

    # Create parent parser for train/eval arguments
    parent_train_evaluate_parser = create_train_and_evaluate_parent_parser()

    # Subparsers
    subparsers = parser.add_subparsers(
        title='commands',
        required=True,
        description='use commands to perform various actions',
        help='additional help for commands'
    )
    create_train_subparser(
        subparsers,
        [parent_parser, parent_train_evaluate_parser]
    )
    create_evaluate_subparser(
        subparsers,
        [parent_parser, parent_train_evaluate_parser]
    )
    create_dataset_subparser(subparsers, [parent_parser])

    return parser


def create_train_and_evaluate_parent_parser():
    """Create parent parser for the `train` and `evaluate` command."""
    parent_parser = argparse.ArgumentParser(
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Optional arguments
    parent_parser.add_argument(
        '--test-data-path',
        type=str,
        default=None,
        help='Path to the development or test dataset in train or '
             'evaluation mode, respectively.'
    )
    parent_parser.add_argument(
        '--models-dir',
        type=str,
        default=None,
        help='Path to the directory where models are saved each epoch. If '
             '`None` then models will not be saved.'
    )
    parent_parser.add_argument(
        '--logs-dir',
        type=str,
        default=None,
        help='Path to the directory where TensorBoard logs are saved each '
             'epoch. If `None` then logs will not be saved.'
    )
    parent_parser.add_argument(
        '--checkpoint-path',
        type=str,
        default=None,
        help='Path to the model checkpoint to be evaluated or to continue '
             'in a training from it.'
    )
    parent_parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='How many subprocesses to use for data loading. `0` means '
             'that the data will be loaded in the main process.'
    )
    parent_parser.add_argument(
        '--print-summary',
        action='store_true',
        help='Whether to print model summary.'
    )

    # Required named arguments
    parser_required_named = parent_parser.add_argument_group(
        'required named arguments'
    )
    parser_required_named.add_argument(
        '--train-data-path',
        required=True,
        type=str,
        help='Path to the training dataset.'
    )
    parser_required_named.add_argument(
        '--model',
        type=arg_map(**model_mapping),
        metavar=f'{{{",".join(key for key in model_mapping.keys())}}}',
        help='Model to be trained.'
    )

    # Hyperparameters
    parser_hparams = parent_parser.add_argument_group(
        'shared hyperparameters'
    )
    parser_hparams.add_argument(
        '--learning-rate',
        type=float,
        default=0.01,
        help='Learning rate.'
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
        '--dropout-prob',
        type=float,
        default=0.5,
        help='Probability of an element to be zeroed.'
    )

    # BiLSTM hyperparameters
    parser_bilstm_hparams = parent_parser.add_argument_group(
        "BiLSTM model's hyperparameters"
    )
    parser_bilstm_hparams.add_argument(
        '--lstm-hidden-size',
        type=int,
        default=256,
        help='Number of neurons in a LSTM layer.'
    )
    parser_bilstm_hparams.add_argument(
        '--max-pooling',
        action='store_true',
        help="whether max-pooling over LSTM's output is enabled or not. "
             "If max-pooling is not enabled, only last LSTM's hidden "
             'states are used.'
    )
    parser_bilstm_hparams.add_argument(
        '--tokenizer',
        type=str,
        default='spacy',
        help='Tokenizer name to be used by PyTorch `get_tokenizer`'
             'function to get instance of `Tokenizer`.'
    )
    parser_bilstm_hparams.add_argument(
        '--tokenizer-language',
        type=str,
        default='en_core_web_sm',
        help='Tokenizer language to be used by PyTorch `get_tokenizer`'
             'function to get instance of `Tokenizer`.'
    )
    parser_bilstm_hparams.add_argument(
        '--vectors',
        type=str,
        default='fasttext.simple.300d',
        choices=['fasttext.simple.300d', 'fasttext.en.300d'],
        help='Vectors name to be loaded by PyTorch `load_vectors` method.'
    )

    # Transformer hyperparameters
    parser_transformer_hparams = parent_parser.add_argument_group(
        "Transformer model's hyperparameters"
    )
    parser_transformer_hparams.add_argument(
        '--pretrained-transformer',
        type=str,
        default=None,
        help='Model name or path of pretrained Transformer model.'
    )

    return parent_parser


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

    # Add execution function
    parser_evaluate.set_defaults(func=evaluate)


def create_dataset_subparser(
    subparsers: argparse._SubParsersAction,
    parents: Optional[List[argparse.ArgumentParser]] = None
) -> None:
    """
    Create parser for the `dataset` command.

    :param subparsers: argparse subparsers where a new parser will be
        added.
    :param parents: subparser parents.
    """
    parser_dataset = subparsers.add_parser(
        'dataset',
        parents=[] if parents is None else parents,
        help='create train, dev and test dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required named arguments
    parser_required_named = parser_dataset.add_argument_group(
        'required named arguments'
    )
    parser_required_named.add_argument(
        '--dataset',
        required=True,
        type=arg_map(**dataset_mapping),
        metavar=f'{{{",".join(key for key in dataset_mapping.keys())}}}',
        help='Name of dataset for appropriate preprocess function.'
    )
    parser_required_named.add_argument(
        '--data-path',
        required=True,
        type=str,
        help='Path to the data.'
    )

    # Dataset specification arguments
    parser_dataset_specs = parser_dataset.add_argument_group(
        'dataset specification arguments'
    )
    parser_dataset_specs.add_argument(
        '--train-size',
        type=int,
        default=80,
        help='Size of train dataset in percentage.'
    )
    parser_dataset_specs.add_argument(
        '--dev-size',
        type=int,
        default=10,
        help='Size of development dataset in percentage.'
    )
    parser_dataset_specs.add_argument(
        '--test-size',
        type=int,
        default=10,
        help='Size of test dataset in percentage.'
    )
    parser_dataset_specs.add_argument(
        '--sanity-size',
        type=int,
        default=0,
        help='If greater than `0` then create dataset for sanity check '
             'with `--sanity-size` elements. Ignore `--train-size`, '
             '`--dev-size` and `--test-size` if set.'
    )
    parser_dataset_specs.add_argument(
        '--shuffle',
        action='store_true',
        help='Whether data should be shuffled before splitting into train, '
             'dev and test datasets.'
    )
    parser_dataset_specs.add_argument(
        '--seed',
        type=int,
        default=13,
        help='Random seed used to create random permutation when `shuffle` '
             'is `True`.'
    )

    # Add execution function
    parser_dataset.set_defaults(func=dataset)


# Define mappings
logging_level_mapping = {
    'critical': logging.CRITICAL,
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}

model_mapping = {
    'baseline': BiLSTMModel,
    'transformer': Transformer
}

dataset_mapping = {
    'contradictory-my-dear-watson': ContradictoryMyDearWatsonDataset,
    'contradictory-my-dear-watson-multilingual':
        ContradictoryMyDearWatsonMultilingualDataset,
    'snli': SNLI
}
