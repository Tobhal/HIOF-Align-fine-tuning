import argparse
from os.path import join as ospj


def qualitative_retrieval_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group('Qualitative retrieval arguments')

    g.add_argument('--model_dir', type=str,
                    default=ospj('saved_models', 'align-fine-tune', 'fold_3_aug_square', '4'),
                    help='Run folder to load the checkpoint + model_args.toml from '
                         '(default: the best ALIGN config).')
    g.add_argument('--checkpoint_name', type=str, default='best.pt',
                    help='Checkpoint filename inside model_dir.')
    g.add_argument('--model_family', type=str, choices=['align', 'clip'], default='align',
                    help='Which architecture model_dir holds a checkpoint for.')

    g.add_argument('--num_queries', type=int, default=10,
                    help='Number of unseen query words to sample from the test/val fold. '
                         'Ignored if --query_words_file is given.')
    g.add_argument('--top_k', type=int, default=5,
                    help='Number of top retrievals to show per query.')
    g.add_argument('--seed', type=int, default=42,
                    help='Random seed for sampling query words (reproducible across runs). '
                         'Ignored if --query_words_file is given.')
    g.add_argument('--query_words_file', type=str, default=None,
                    help='Optional path to a newline-separated list of exact query words to '
                         'use instead of sampling -- use this to guarantee a second run (e.g. '
                         'a different model family) queries the *same* words as an earlier run, '
                         'since two runs can have slightly different usable image pools and so '
                         'the same --seed is not guaranteed to reproduce the same sample.')

    g.add_argument('--out_dir', type=str, default=ospj('results', 'qualitative'),
                    help='Directory to write per-query figures + summary.csv into.')

    return parser
