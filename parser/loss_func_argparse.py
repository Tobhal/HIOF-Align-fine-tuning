import argparse

def loss_func_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    loss_parser = parser.add_argument_group('Loss function arguments')
    loss_parser.add_argument('--loss_func', choices=['triplet', 'contrastive', 'simple', 'supcon'],
                             default='supcon',
                             help='loss function for training')
    return parser
