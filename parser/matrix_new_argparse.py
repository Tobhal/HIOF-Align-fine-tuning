import argparse
from os.path import join as ospj

import argparse
from os.path import join as ospj

def matrix_new_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group('Matrix arguments')

    g.add_argument('--nums', nargs='+', type=int, default=[1],
                   help='Checkpoint numbers to process (e.g., --nums 1 2 3).')

    # Which matrix to compute
    g.add_argument('--evaluate', choices=['image', 'text'], default='image',
                   help='Compute image–image or text–text cosine matrix.')

    # Where to load weights from
    g.add_argument('--model_source', choices=['fine-tuned', 'pretrained'], default='fine-tuned',
                   help='Use fine-tuned checkpoint or base pretrained weights.')

    g.add_argument('--checkpoint_name', type=str, default='best.pt',
                   help='Checkpoint filename when model_source=fine-tuned.')

    # Optional: heatmap controls (used by save_heatmap)
    g.add_argument('--heatmap', action='store_true',
                   help='Save a PNG heatmap next to the CSV.')
    g.add_argument('--downsample_block', type=int, default=1,
                   help='Avg-pool factor for heatmap (1 = none).')
    g.add_argument('--cell_px', type=int, default=3,
                   help='Pixels per cell in the heatmap.')
    g.add_argument('--max_width_px', type=int, default=1800,
                   help='Max heatmap width in pixels.')
    g.add_argument('--cmap', type=str, default='coolwarm',
                   help='Set the cmap for the heatmap')

    return parser