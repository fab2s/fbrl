import argparse

from fbrl.data import (generate_dataset, generate_test, generate_bigram_dataset,
                       generate_bigram_test, generate_word_dataset, generate_word_test)
from fbrl.train import (train_model, check_attention, train_bigram_model,
                         check_bigram_attention, train_word_model)
from fbrl.evaluate import (test_model, visualize_model, generate_atlas, test_bigram_model,
                           generate_bigram_atlas, test_word_model, generate_word_atlas,
                           test_word_isolation)


def _parse_patch_size(s):
    """Parse patch size: '12' -> 12 (int), '12,18' -> (12, 18) (tuple)."""
    parts = s.split(',')
    if len(parts) == 1:
        return int(parts[0])
    return tuple(int(x) for x in parts)


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
    train_parser.add_argument('--diversity_vy', type=float, default=1.0,
                              help='Vertical diversity multiplier (1.5 = 50%% stronger vertical repulsion)')
    train_parser.add_argument('--n_scan_glimpses', type=int, default=0,
                              help='Scan-phase glimpses (0=no scan, 3=recommended)')
    train_parser.add_argument('--scan_patch_size', default='12,18',
                              help='Scan patch size as H,W (default: 12,18)')
    train_parser.add_argument('--scan_vy', type=float, default=0.3,
                              help='Scan diversity VY (<1 = horizontal spread)')
    train_parser.add_argument('--scan_guide_weight', type=float, default=None,
                              help='Scan guide weight (defaults to --guide_weight)')
    train_parser.add_argument('--content_weight', type=float, default=0.5,
                              help='Content detection BCE weight (0=off)')

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
    chk_parser.add_argument('--diversity_vy', type=float, default=1.0,
                            help='Vertical diversity multiplier (1.5 = 50%% stronger vertical repulsion)')

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
    train_bi_parser.add_argument('--n_scan_glimpses', type=int, default=5,
                                 help='Number of scan-phase glimpses (wide patches)')
    train_bi_parser.add_argument('--n_read_glimpses', type=int, default=6,
                                 help='Number of read-phase glimpses (focused patches)')
    train_bi_parser.add_argument('--scan_patch_size', default='12,18',
                                 help='Scan patch size as H,W (default: 12,18)')
    train_bi_parser.add_argument('--read_patch_size', type=int, default=12,
                                 help='Read patch size (square, default: 12)')
    train_bi_parser.add_argument('--n_scales', type=int, default=1)
    train_bi_parser.add_argument('--device', default='auto',
                                 choices=['auto', 'cpu', 'cuda'])
    train_bi_parser.add_argument('--resume', default=None)
    train_bi_parser.add_argument('--diversity_weight', type=float, default=1.0)
    train_bi_parser.add_argument('--diversity_sigma', type=float, default=0.1)
    train_bi_parser.add_argument('--guide_weight', type=float, default=8.0)
    train_bi_parser.add_argument('--scan_guide_weight', type=float, default=None,
                                 help='Scan-phase guide weight (defaults to --guide_weight)')
    train_bi_parser.add_argument('--blur_sigma_ratio', type=float, default=0.16)
    train_bi_parser.add_argument('--batch_size', type=int, default=32)
    train_bi_parser.add_argument('--scaffold_epochs', type=int, default=None,
                                 help='Explicit scaffold epoch count (overrides --scaffold_ratio)')
    train_bi_parser.add_argument('--scaffold_ratio', type=float, default=0.67,
                                 help='Scaffold phase as fraction of total epochs '
                                      '(default 0.67 = 67%%). Ignored if --scaffold_epochs set.')
    train_bi_parser.add_argument('--scaffold_floor', type=float, default=0.0,
                                 help='Minimum scaffold weight after annealing '
                                      '(0.05 keeps gentle spatial pressure)')
    train_bi_parser.add_argument('--transfer', default=None,
                                 help='Path to single-letter .pth/.pth.gz for transfer learning')
    train_bi_parser.add_argument('--mask_weight', type=float, default=0.5,
                                 help='Weight for masked-half auxiliary loss (0=disabled)')
    train_bi_parser.add_argument('--scan_vy', type=float, default=0.3,
                                 help='Scan diversity VY (<1 = horizontal spread, default 0.3)')
    train_bi_parser.add_argument('--read_vy', type=float, default=1.5,
                                 help='Read diversity VY (>1 = vertical exploration, default 1.5)')
    train_bi_parser.add_argument('--edge_weight', type=float, default=0.0,
                                 help='Edge exploration weight — pushes scan fixations toward image sides (0=off)')

    chk_bi_parser = subparsers.add_parser('check_bigram_attention')
    chk_bi_parser.add_argument('--data_dir', required=True)
    chk_bi_parser.add_argument('--n_epochs', type=int, default=10)
    chk_bi_parser.add_argument('--n_scan_glimpses', type=int, default=5)
    chk_bi_parser.add_argument('--n_read_glimpses', type=int, default=6)
    chk_bi_parser.add_argument('--scan_patch_size', default='12,18')
    chk_bi_parser.add_argument('--read_patch_size', type=int, default=12)
    chk_bi_parser.add_argument('--n_scales', type=int, default=1)
    chk_bi_parser.add_argument('--device', default='auto',
                                choices=['auto', 'cpu', 'cuda'])
    chk_bi_parser.add_argument('--guide_weight', type=float, default=8.0)
    chk_bi_parser.add_argument('--blur_sigma_ratio', type=float, default=0.16)
    chk_bi_parser.add_argument('--diversity_weight', type=float, default=1.0)
    chk_bi_parser.add_argument('--diversity_sigma', type=float, default=0.1)
    chk_bi_parser.add_argument('--diversity_vy', type=float, default=1.0,
                                help='Vertical diversity multiplier (1.5 = 50%% stronger vertical repulsion)')

    test_bi_parser = subparsers.add_parser('test_bigrams')
    test_bi_parser.add_argument('--model_dir', required=True)
    test_bi_parser.add_argument('--test_data_dir', required=True)
    test_bi_parser.add_argument('--output_dir', default='bigram_results')
    test_bi_parser.add_argument('--device', default='auto',
                                 choices=['auto', 'cpu', 'cuda'])

    bi_atlas_parser = subparsers.add_parser('bigram_atlas')
    bi_atlas_parser.add_argument('--model_dir', required=True)
    bi_atlas_parser.add_argument('--test_data_dir', required=True)
    bi_atlas_parser.add_argument('--output', default='data/bigram_atlas.html')
    bi_atlas_parser.add_argument('--device', default='auto',
                                  choices=['auto', 'cpu', 'cuda'])

    # --- Word subcommands ---
    gen_w_parser = subparsers.add_parser('generate_words')
    gen_w_parser.add_argument('--num_variants', type=int, default=20)
    gen_w_parser.add_argument('--noise_level', type=float, default=0.01)
    gen_w_parser.add_argument('--output_dir', default='data/words')
    gen_w_parser.add_argument('--fonts', default='default',
                              help='Font spec: "all", "default", or comma-separated names')

    gen_w_test_parser = subparsers.add_parser('generate_words_test')
    gen_w_test_parser.add_argument('--output_dir', default='data/word_test')
    gen_w_test_parser.add_argument('--fonts', default='default',
                                   help='Font spec: "all", "default", or comma-separated names')

    train_w_parser = subparsers.add_parser('train_words')
    train_w_parser.add_argument('--data_dir', required=True)
    train_w_parser.add_argument('--epochs', type=int, default=200)
    train_w_parser.add_argument('--save_dir', default='word_models')
    train_w_parser.add_argument('--checkpoint_interval', type=int, default=10)
    train_w_parser.add_argument('--n_scan_glimpses', type=int, default=8,
                                help='Number of scan-phase glimpses (prescribed x)')
    train_w_parser.add_argument('--n_read_glimpses', type=int, default=12,
                                help='Number of read-phase glimpses (free)')
    train_w_parser.add_argument('--scan_patch_size', default='12,18',
                                help='Scan patch size as H,W (default: 12,18)')
    train_w_parser.add_argument('--read_patch_size', type=int, default=12,
                                help='Read patch size (square, default: 12)')
    train_w_parser.add_argument('--n_scales', type=int, default=1)
    train_w_parser.add_argument('--n_positions', type=int, default=4,
                                help='Number of letter positions (default: 4)')
    train_w_parser.add_argument('--device', default='auto',
                                choices=['auto', 'cpu', 'cuda'])
    train_w_parser.add_argument('--resume', default=None)
    train_w_parser.add_argument('--diversity_weight', type=float, default=1.0)
    train_w_parser.add_argument('--diversity_sigma', type=float, default=0.1)
    train_w_parser.add_argument('--guide_weight', type=float, default=8.0)
    train_w_parser.add_argument('--scan_guide_weight', type=float, default=None,
                                help='Scan-phase guide weight (defaults to --guide_weight)')
    train_w_parser.add_argument('--blur_sigma_ratio', type=float, default=0.16)
    train_w_parser.add_argument('--batch_size', type=int, default=32)
    train_w_parser.add_argument('--scaffold_epochs', type=int, default=None,
                                help='Explicit scaffold epoch count (overrides --scaffold_ratio)')
    train_w_parser.add_argument('--scaffold_ratio', type=float, default=0.67,
                                help='Scaffold phase as fraction of total epochs')
    train_w_parser.add_argument('--scaffold_floor', type=float, default=0.0,
                                help='Minimum scaffold weight after annealing')
    train_w_parser.add_argument('--transfer', default=None,
                                help='Path to single-letter .pth/.pth.gz for transfer learning')
    train_w_parser.add_argument('--content_weight', type=float, default=0.5,
                                help='Weight for content detection BCE loss (0=disabled)')
    train_w_parser.add_argument('--isolation_weight', type=float, default=0.5,
                                help='Weight for isolation mask loss — masks 3 of 4 letters, '
                                     'forces single-letter fixation (0=disabled)')
    train_w_parser.add_argument('--scan_vy', type=float, default=0.3,
                                help='Scan diversity VY (<1 = horizontal spread)')
    train_w_parser.add_argument('--read_vy', type=float, default=1.5,
                                help='Read diversity VY (>1 = vertical exploration)')
    train_w_parser.add_argument('--edge_weight', type=float, default=0.0,
                                help='Edge exploration weight (0=off, prescribed x makes this unnecessary)')
    train_w_parser.add_argument('--isolation_data_dir', default=None,
                                help='Path to 128x128 single-letter data for isolation testing '
                                     '(e.g. data/letters). When set, replaces mask-based isolation.')
    train_w_parser.add_argument('--isolation_random_prob', type=float, default=0.0,
                                help='Probability of substituting a random letter in isolation (0-1)')
    train_w_parser.add_argument('--multi_head', action='store_true', default=False,
                                help='Use 3 separate optimizers for attention/classification/reconstruction')
    train_w_parser.add_argument('--amp', action='store_true', default=False,
                                help='Enable Automatic Mixed Precision (FP16) — halves VRAM usage')

    test_w_parser = subparsers.add_parser('test_words')
    test_w_parser.add_argument('--model_dir', required=True)
    test_w_parser.add_argument('--test_data_dir', required=True)
    test_w_parser.add_argument('--output_dir', default='word_results')
    test_w_parser.add_argument('--device', default='auto',
                               choices=['auto', 'cpu', 'cuda'])

    iso_test_parser = subparsers.add_parser('test_word_isolation')
    iso_test_parser.add_argument('--model_dir', required=True)
    iso_test_parser.add_argument('--test_data_dir', required=True)
    iso_test_parser.add_argument('--output_dir', default='word_results')
    iso_test_parser.add_argument('--device', default='auto',
                                  choices=['auto', 'cpu', 'cuda'])

    w_atlas_parser = subparsers.add_parser('word_atlas')
    w_atlas_parser.add_argument('--model_dir', required=True)
    w_atlas_parser.add_argument('--test_data_dir', required=True)
    w_atlas_parser.add_argument('--output', default='data/word_atlas.html')
    w_atlas_parser.add_argument('--device', default='auto',
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
        scan_ps = _parse_patch_size(args.scan_patch_size)
        train_model(args.data_dir, args.epochs, args.resume, args.save_dir,
                    args.checkpoint_interval, n_glimpses=args.n_glimpses,
                    patch_size=args.patch_size, n_scales=args.n_scales,
                    device=args.device,
                    diversity_weight=args.diversity_weight,
                    diversity_sigma=args.diversity_sigma,
                    diversity_vy=args.diversity_vy,
                    recode_weight=args.recode_weight,
                    guide_weight=args.guide_weight,
                    blur_sigma_ratio=args.blur_sigma_ratio,
                    batch_size=args.batch_size,
                    n_scan_glimpses=args.n_scan_glimpses,
                    scan_patch_size=scan_ps,
                    scan_vy=args.scan_vy,
                    scan_guide_weight=args.scan_guide_weight,
                    content_weight=args.content_weight)

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
                        diversity_sigma=args.diversity_sigma,
                        diversity_vy=args.diversity_vy)

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
        # Resolve scaffold duration: explicit --scaffold_epochs wins, else use ratio
        scaffold_ep = (args.scaffold_epochs if args.scaffold_epochs is not None
                       else int(args.scaffold_ratio * args.epochs))
        scan_ps = _parse_patch_size(args.scan_patch_size)
        train_bigram_model(args.data_dir, args.epochs, args.resume, args.save_dir,
                           args.checkpoint_interval,
                           n_scan_glimpses=args.n_scan_glimpses,
                           n_read_glimpses=args.n_read_glimpses,
                           scan_patch_size=scan_ps,
                           read_patch_size=args.read_patch_size,
                           n_scales=args.n_scales,
                           device=args.device,
                           diversity_weight=args.diversity_weight,
                           diversity_sigma=args.diversity_sigma,
                           scan_vy=args.scan_vy,
                           read_vy=args.read_vy,
                           guide_weight=args.guide_weight,
                           scan_guide_weight=args.scan_guide_weight,
                           blur_sigma_ratio=args.blur_sigma_ratio,
                           batch_size=args.batch_size,
                           scaffold_epochs=scaffold_ep,
                           scaffold_floor=args.scaffold_floor,
                           transfer_from=args.transfer,
                           mask_weight=args.mask_weight,
                           edge_weight=args.edge_weight)

    elif args.command == 'check_bigram_attention':
        scan_ps = _parse_patch_size(args.scan_patch_size)
        check_bigram_attention(args.data_dir, n_epochs=args.n_epochs,
                               n_scan_glimpses=args.n_scan_glimpses,
                               n_read_glimpses=args.n_read_glimpses,
                               scan_patch_size=scan_ps,
                               read_patch_size=args.read_patch_size,
                               n_scales=args.n_scales, device=args.device,
                               guide_weight=args.guide_weight,
                               blur_sigma_ratio=args.blur_sigma_ratio,
                               diversity_weight=args.diversity_weight,
                               diversity_sigma=args.diversity_sigma,
                               diversity_vy=args.diversity_vy)

    elif args.command == 'test_bigrams':
        test_bigram_model(args.model_dir, args.test_data_dir, args.output_dir,
                          device=args.device)

    elif args.command == 'bigram_atlas':
        generate_bigram_atlas(args.model_dir, args.test_data_dir, args.output,
                              device=args.device)

    # --- Word commands ---
    elif args.command == 'generate_words':
        generate_word_dataset(args.output_dir, args.noise_level, args.num_variants,
                               font_spec=args.fonts)

    elif args.command == 'generate_words_test':
        generate_word_test(args.output_dir, font_spec=args.fonts)

    elif args.command == 'train_words':
        scaffold_ep = (args.scaffold_epochs if args.scaffold_epochs is not None
                       else int(args.scaffold_ratio * args.epochs))
        scan_ps = _parse_patch_size(args.scan_patch_size)
        train_word_model(args.data_dir, args.epochs, args.resume, args.save_dir,
                          args.checkpoint_interval,
                          n_scan_glimpses=args.n_scan_glimpses,
                          n_read_glimpses=args.n_read_glimpses,
                          scan_patch_size=scan_ps,
                          read_patch_size=args.read_patch_size,
                          n_scales=args.n_scales,
                          n_positions=args.n_positions,
                          device=args.device,
                          diversity_weight=args.diversity_weight,
                          diversity_sigma=args.diversity_sigma,
                          scan_vy=args.scan_vy,
                          read_vy=args.read_vy,
                          guide_weight=args.guide_weight,
                          scan_guide_weight=args.scan_guide_weight,
                          blur_sigma_ratio=args.blur_sigma_ratio,
                          batch_size=args.batch_size,
                          scaffold_epochs=scaffold_ep,
                          scaffold_floor=args.scaffold_floor,
                          transfer_from=args.transfer,
                          content_weight=args.content_weight,
                          isolation_weight=args.isolation_weight,
                          edge_weight=args.edge_weight,
                          isolation_data_dir=args.isolation_data_dir,
                          isolation_random_prob=args.isolation_random_prob,
                          multi_head=args.multi_head,
                          amp=args.amp)

    elif args.command == 'test_words':
        test_word_model(args.model_dir, args.test_data_dir, args.output_dir,
                         device=args.device)

    elif args.command == 'test_word_isolation':
        test_word_isolation(args.model_dir, args.test_data_dir, args.output_dir,
                             device=args.device)

    elif args.command == 'word_atlas':
        generate_word_atlas(args.model_dir, args.test_data_dir, args.output,
                             device=args.device)
