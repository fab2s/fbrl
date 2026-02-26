import argparse

from fbrl.data import generate_dataset, generate_test, generate_bigram_dataset, generate_bigram_test
from fbrl.train import train_model, check_attention, train_bigram_model, check_bigram_attention
from fbrl.evaluate import test_model, visualize_model, generate_atlas, test_bigram_model


def _parse_letters(letters_str):
    if letters_str == 'A-Z':
        return [chr(i) for i in range(65, 91)]
    if letters_str == 'a-z':
        return [chr(i) for i in range(97, 123)]
    if letters_str in ('Aa-Zz', 'A-Za-z'):
        return [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
    return list(letters_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vision-Only Letter Training Pipeline')
    subparsers = parser.add_subparsers(dest='command', required=True)

    gen_parser = subparsers.add_parser('generate')
    gen_parser.add_argument('--letters', default='Aa-Zz')
    gen_parser.add_argument('--num_variants', type=int, default=20)
    gen_parser.add_argument('--noise_level', type=float, default=0.01)
    gen_parser.add_argument('--output_dir', default='data/letters')
    gen_parser.add_argument('--fonts', default='all',
                            help='Font spec: "all", "default", or comma-separated names')

    gentest_parser = subparsers.add_parser('generate_test')
    gentest_parser.add_argument('--letters', default='Aa-Zz')
    gentest_parser.add_argument('--output_dir', default='data/test')
    gentest_parser.add_argument('--fonts', default='all',
                                help='Font spec: "all", "default", or comma-separated names')

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--data_dir', required=True)
    train_parser.add_argument('--epochs', type=int, default=200)
    train_parser.add_argument('--save_dir', default='models')
    train_parser.add_argument('--checkpoint_interval', type=int, default=10)
    train_parser.add_argument('--n_glimpses', type=int, default=10)
    train_parser.add_argument('--patch_size', type=int, default=12)
    train_parser.add_argument('--n_scales', type=int, default=1)
    train_parser.add_argument('--device', default='auto',
                              choices=['auto', 'cpu', 'cuda'])
    train_parser.add_argument('--resume', default=None)
    train_parser.add_argument('--diversity_weight', type=float, default=1.0,
                              help='Weight for fixation diversity loss (0=off)')
    train_parser.add_argument('--diversity_sigma', type=float, default=0.1,
                              help='Repulsion radius in normalized coords (0.1=10%% of image)')
    train_parser.add_argument('--recode_weight', type=float, default=1.0,
                              help='Weight for recode loss (0=off)')
    train_parser.add_argument('--guide_weight', type=float, default=8.0,
                              help='Weight for attention guide loss')
    train_parser.add_argument('--blur_sigma_ratio', type=float, default=0.16,
                              help='Blur sigma as fraction of image size (0.16=proven default)')
    train_parser.add_argument('--batch_size', type=int, default=52,
                              help='Training batch size (default 52)')

    test_parser = subparsers.add_parser('test')
    test_parser.add_argument('--model_dir', required=True)
    test_parser.add_argument('--test_data_dir', required=True)
    test_parser.add_argument('--output_dir', default='results')
    test_parser.add_argument('--device', default='auto',
                             choices=['auto', 'cpu', 'cuda'])

    viz_parser = subparsers.add_parser('visualize')
    viz_parser.add_argument('--model_dir', required=True)
    viz_parser.add_argument('--data_dir', required=True)
    viz_parser.add_argument('--output_dir', default='visualizations')
    viz_parser.add_argument('--device', default='auto',
                            choices=['auto', 'cpu', 'cuda'])

    atlas_parser = subparsers.add_parser('atlas')
    atlas_parser.add_argument('--model_dir', required=True)
    atlas_parser.add_argument('--test_data_dir', required=True)
    atlas_parser.add_argument('--output', default='data/atlas.html')
    atlas_parser.add_argument('--device', default='auto',
                              choices=['auto', 'cpu', 'cuda'])

    chk_parser = subparsers.add_parser('check_attention')
    chk_parser.add_argument('--data_dir', required=True)
    chk_parser.add_argument('--n_epochs', type=int, default=10)
    chk_parser.add_argument('--n_glimpses', type=int, default=10)
    chk_parser.add_argument('--patch_size', type=int, default=12)
    chk_parser.add_argument('--n_scales', type=int, default=1)
    chk_parser.add_argument('--device', default='auto',
                            choices=['auto', 'cpu', 'cuda'])
    chk_parser.add_argument('--guide_weight', type=float, default=8.0)
    chk_parser.add_argument('--blur_sigma_ratio', type=float, default=0.16)
    chk_parser.add_argument('--diversity_weight', type=float, default=1.0)
    chk_parser.add_argument('--diversity_sigma', type=float, default=0.1)

    compress_parser = subparsers.add_parser('compress_model')
    compress_parser.add_argument('--input', required=True)
    compress_parser.add_argument('--output', required=True)

    # --- Bigram subcommands ---
    gen_bi_parser = subparsers.add_parser('generate_bigrams')
    gen_bi_parser.add_argument('--num_variants', type=int, default=20)
    gen_bi_parser.add_argument('--noise_level', type=float, default=0.01)
    gen_bi_parser.add_argument('--output_dir', default='data/bigrams')
    gen_bi_parser.add_argument('--fonts', default='default',
                               help='Font spec: "all", "default", or comma-separated names')

    gen_bi_test_parser = subparsers.add_parser('generate_bigrams_test')
    gen_bi_test_parser.add_argument('--output_dir', default='data/bigram_test')
    gen_bi_test_parser.add_argument('--fonts', default='default',
                                    help='Font spec: "all", "default", or comma-separated names')

    train_bi_parser = subparsers.add_parser('train_bigrams')
    train_bi_parser.add_argument('--data_dir', required=True)
    train_bi_parser.add_argument('--epochs', type=int, default=100)
    train_bi_parser.add_argument('--save_dir', default='bigram_models')
    train_bi_parser.add_argument('--checkpoint_interval', type=int, default=10)
    train_bi_parser.add_argument('--n_glimpses', type=int, default=15)
    train_bi_parser.add_argument('--patch_size', type=int, default=12)
    train_bi_parser.add_argument('--n_scales', type=int, default=1)
    train_bi_parser.add_argument('--device', default='auto',
                                 choices=['auto', 'cpu', 'cuda'])
    train_bi_parser.add_argument('--resume', default=None)
    train_bi_parser.add_argument('--diversity_weight', type=float, default=1.0)
    train_bi_parser.add_argument('--diversity_sigma', type=float, default=0.1)
    train_bi_parser.add_argument('--guide_weight', type=float, default=8.0)
    train_bi_parser.add_argument('--blur_sigma_ratio', type=float, default=0.16)
    train_bi_parser.add_argument('--batch_size', type=int, default=32)

    chk_bi_parser = subparsers.add_parser('check_bigram_attention')
    chk_bi_parser.add_argument('--data_dir', required=True)
    chk_bi_parser.add_argument('--n_epochs', type=int, default=10)
    chk_bi_parser.add_argument('--n_glimpses', type=int, default=15)
    chk_bi_parser.add_argument('--patch_size', type=int, default=12)
    chk_bi_parser.add_argument('--n_scales', type=int, default=1)
    chk_bi_parser.add_argument('--device', default='auto',
                                choices=['auto', 'cpu', 'cuda'])
    chk_bi_parser.add_argument('--guide_weight', type=float, default=8.0)
    chk_bi_parser.add_argument('--blur_sigma_ratio', type=float, default=0.16)
    chk_bi_parser.add_argument('--diversity_weight', type=float, default=1.0)
    chk_bi_parser.add_argument('--diversity_sigma', type=float, default=0.1)

    test_bi_parser = subparsers.add_parser('test_bigrams')
    test_bi_parser.add_argument('--model_dir', required=True)
    test_bi_parser.add_argument('--test_data_dir', required=True)
    test_bi_parser.add_argument('--output_dir', default='bigram_results')
    test_bi_parser.add_argument('--device', default='auto',
                                 choices=['auto', 'cpu', 'cuda'])

    args = parser.parse_args()

    if args.command == 'generate':
        letters = _parse_letters(args.letters)
        generate_dataset(letters, args.output_dir, args.noise_level, args.num_variants,
                         font_spec=args.fonts)

    elif args.command == 'generate_test':
        letters = _parse_letters(args.letters)
        generate_test(letters, args.output_dir, font_spec=args.fonts)

    elif args.command == 'train':
        train_model(args.data_dir, args.epochs, args.resume, args.save_dir,
                    args.checkpoint_interval, n_glimpses=args.n_glimpses,
                    patch_size=args.patch_size, n_scales=args.n_scales,
                    device=args.device,
                    diversity_weight=args.diversity_weight,
                    diversity_sigma=args.diversity_sigma,
                    recode_weight=args.recode_weight,
                    guide_weight=args.guide_weight,
                    blur_sigma_ratio=args.blur_sigma_ratio,
                    batch_size=args.batch_size)

    elif args.command == 'test':
        test_model(args.model_dir, args.test_data_dir, args.output_dir,
                   device=args.device)

    elif args.command == 'visualize':
        visualize_model(args.model_dir, args.data_dir, args.output_dir,
                        device=args.device)

    elif args.command == 'atlas':
        generate_atlas(args.model_dir, args.test_data_dir, args.output,
                       device=args.device)

    elif args.command == 'check_attention':
        check_attention(args.data_dir, n_epochs=args.n_epochs,
                        n_glimpses=args.n_glimpses, patch_size=args.patch_size,
                        n_scales=args.n_scales, device=args.device,
                        guide_weight=args.guide_weight,
                        blur_sigma_ratio=args.blur_sigma_ratio,
                        diversity_weight=args.diversity_weight,
                        diversity_sigma=args.diversity_sigma)

    elif args.command == 'compress_model':
        import torch, gzip, io, os
        ckpt = torch.load(args.input, map_location='cpu', weights_only=False)
        ckpt['model'] = {k: v.half() for k, v in ckpt['model'].items()}
        buf = io.BytesIO()
        torch.save(ckpt, buf)
        raw = buf.getvalue()
        with gzip.open(args.output, 'wb') as f:
            f.write(raw)
        orig_mb = os.path.getsize(args.input) / 1048576
        comp_mb = os.path.getsize(args.output) / 1048576
        print(f"fp16+gzip: {orig_mb:.0f}MB -> {comp_mb:.0f}MB ({comp_mb/orig_mb:.0%})")

    # --- Bigram commands ---
    elif args.command == 'generate_bigrams':
        generate_bigram_dataset(args.output_dir, args.noise_level, args.num_variants,
                                font_spec=args.fonts)

    elif args.command == 'generate_bigrams_test':
        generate_bigram_test(args.output_dir, font_spec=args.fonts)

    elif args.command == 'train_bigrams':
        train_bigram_model(args.data_dir, args.epochs, args.resume, args.save_dir,
                           args.checkpoint_interval, n_glimpses=args.n_glimpses,
                           patch_size=args.patch_size, n_scales=args.n_scales,
                           device=args.device,
                           diversity_weight=args.diversity_weight,
                           diversity_sigma=args.diversity_sigma,
                           guide_weight=args.guide_weight,
                           blur_sigma_ratio=args.blur_sigma_ratio,
                           batch_size=args.batch_size)

    elif args.command == 'check_bigram_attention':
        check_bigram_attention(args.data_dir, n_epochs=args.n_epochs,
                               n_glimpses=args.n_glimpses, patch_size=args.patch_size,
                               n_scales=args.n_scales, device=args.device,
                               guide_weight=args.guide_weight,
                               blur_sigma_ratio=args.blur_sigma_ratio,
                               diversity_weight=args.diversity_weight,
                               diversity_sigma=args.diversity_sigma)

    elif args.command == 'test_bigrams':
        test_bigram_model(args.model_dir, args.test_data_dir, args.output_dir,
                          device=args.device)
