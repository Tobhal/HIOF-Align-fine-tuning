import argparse

def training_common_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    common_parser = parser.add_argument_group('Common training arguments')
    
    common_parser.add_argument('--accumulation_steps', type=int, default=4,
                                help='number of steps to accumulate gradients')

    common_parser.add_argument('--description', choices=['word', 'description', 'phosc_number', 'description_long'],
                                default='word', help='description to use for fine-tuning')
    
    common_parser.add_argument("--mse_weight", type=float, default=0.1)
    common_parser.add_argument("--head_hidden_dim", type=int, default=1024)
    common_parser.add_argument("--phosc_image_size", type=int, default=224)

    return parser
