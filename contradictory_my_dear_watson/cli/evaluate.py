from argparse import Namespace

from contradictory_my_dear_watson.runner import Runner


def evaluate(args: Namespace) -> None:
    """
    Evaluate a trained model.

    :param args: evaluate command's arguments.
    """
    runner = Runner(
        model_class=args.model,
        train_data_path=args.train_data_path,
        test_data_path=args.test_data_path,
        models_dir=args.models_dir,
        logs_dir=args.logs_dir,
        checkpoint_path=args.checkpoint_path,
        num_workers=args.num_workers,
        print_summary=args.print_summary,
        **{
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'dropout_prob': args.dropout_prob,
            'tokenizer': args.tokenizer,
            'tokenizer_language': args.tokenizer_language,
            'vectors': args.vectors,
            'lstm_hidden_size': args.lstm_hidden_size,
            'max_pooling': args.max_pooling,
            'pretrained_transformer': args.pretrained_transformer
        }
    )
    runner.evaluate()
