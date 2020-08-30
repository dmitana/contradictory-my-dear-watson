from contradictory_my_dear_watson.cli.parser import create_argument_parser


if __name__ == '__main__':
    parser = create_argument_parser()

    args = parser.parse_args()
    args.func(args)
