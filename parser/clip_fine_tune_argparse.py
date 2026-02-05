import argparse
from os.path import join as ospj


def clip_fine_tune_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    clip_fine_tune_parser = parser.add_argument_group('CLIP fine-tune arguments')

    # clip_fine_tune_parser.add_argument('--clip_fine_tune_config', type=str, default=ospj('configs', 'clip', 'fine-tune.yml'), help='Path to the clip fine-tune configuration file')

    # clip_fine_tune_parser.add_argument('--no_tqdm_bar', action='store_true', default=False, help='Show tqdm progress bar or not')

    return parser