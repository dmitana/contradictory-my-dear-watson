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
                    'problem.'
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
        'train', help='train a new model'
    )
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
