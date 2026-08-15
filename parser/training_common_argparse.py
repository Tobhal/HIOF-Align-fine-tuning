import argparse

def training_common_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    common_parser = parser.add_argument_group('Common training arguments')
    
    common_parser.add_argument('--accumulation_steps', type=int, default=4,
                                help='number of steps to accumulate gradients')

    common_parser.add_argument('--description', choices=['word', 'description', 'phosc_number', 'description_long'],
                                default='word', help='description to use for fine-tuning')

    common_parser.add_argument('--shuffle_descriptions', action='store_true', default=False,
                                help='Negative control: randomly permute the description/prompt list within '
                                     'each batch before feeding it to the model, so image<->text pairing is '
                                     'broken while the text side keeps the same distribution of real prompts. '
                                     'Everything else (images, words, loss, hyperparameters) stays identical.')

    common_parser.add_argument("--mse_weight", type=float, default=0.1)
    common_parser.add_argument("--head_hidden_dim", type=int, default=1024)
    common_parser.add_argument("--phosc_image_size", type=int, default=224)

    return parser
