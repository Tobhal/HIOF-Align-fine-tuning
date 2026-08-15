import argparse


def retrieval_metrics_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group('Retrieval metrics arguments')

    g.add_argument('--align_nums', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                    help='ALIGN run numbers (under saved_models/<align_save_name>/<split_name>/) to evaluate.')
    g.add_argument('--clip_nums', nargs='+', type=int, default=[1, 2, 4, 5, 6, 7, 8],
                    help='CLIP run numbers (under saved_models/<clip_save_name>/<split_name>/) to evaluate.')

    g.add_argument('--align_save_name', type=str, default='align-fine-tune',
                    help='Folder name under save_dir holding ALIGN runs.')
    g.add_argument('--clip_save_name', type=str, default='clip-fine-tune',
                    help='Folder name under save_dir holding CLIP runs.')

    g.add_argument('--skip_align', action='store_true', help='Skip all ALIGN runs.')
    g.add_argument('--skip_clip', action='store_true', help='Skip all CLIP runs.')
    g.add_argument('--skip_qbe', action='store_true',
                    help='Skip image->image (QbE) retrieval metrics, only compute QbS (text->image).')

    g.add_argument('--recall_ks', nargs='+', type=int, default=[1, 5, 10],
                    help='k values for Recall@k.')
    g.add_argument('--checkpoint_name', type=str, default='best.pt',
                    help='Checkpoint filename to load from each run folder.')

    g.add_argument('--out_dir', type=str, default='results',
                    help='Directory to write the consolidated retrieval_metrics.csv/.md into.')
    g.add_argument('--write_per_run_report', action='store_true', default=True,
                    help='Also write a matrix_{num}_report.txt next to each checkpoint (same convention as matrix_new.py).')

    return parser
