from argparse import Namespace


def dataset(args: Namespace) -> None:
    """
    Create train, dev and test dataset.

    :param args: dataset command's arguments.
    """
    args.dataset(
        args.data_path,
        args.train_size,
        args.dev_size,
        args.test_size,
        args.sanity_size,
        args.shuffle,
        args.seed
    )()
