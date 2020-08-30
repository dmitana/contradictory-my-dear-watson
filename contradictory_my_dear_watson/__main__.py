from contradictory_my_dear_watson.cli.parser import create_argument_parser
from contradictory_my_dear_watson.utils.logging import initialize_logger


if __name__ == '__main__':
    parser = create_argument_parser()

    args = parser.parse_args()
    initialize_logger(
        name='contradictory_my_dear_watson',
        logging_level=args.logging_level
    )
    args.func(args)
