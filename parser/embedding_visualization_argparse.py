import argparse
from os.path import join as ospj


def embedding_visualization_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group('Embedding visualization arguments')

    g.add_argument('--finetuned_model_dir', type=str,
                    default=ospj('saved_models', 'align-fine-tune', 'fold_3_aug_square', '4'),
                    help='Run folder to load the fine-tuned checkpoint from (default: the best ALIGN config).')
    g.add_argument('--checkpoint_name', type=str, default='best.pt')

    g.add_argument('--highlight_words', nargs='+', type=str, default=None,
                    help='Words to draw in distinct colors (max 3, for CVD-safe scatter colors -- '
                         'see results/embeddings/README.md). Everything else is plotted as a muted '
                         '"other" gray. Defaults to 3 words already discussed in the qualitative '
                         'write-up (results/qualitative_writeup.md) for continuity.')

    g.add_argument('--tsne_perplexity', type=float, default=30.0)
    g.add_argument('--seed', type=int, default=42)

    g.add_argument('--out_dir', type=str, default=ospj('results', 'embeddings'))

    return parser
