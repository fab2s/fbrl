import argparse

from fbrl.data import (generate_dataset, generate_test, generate_bigram_dataset,
                       generate_bigram_test, generate_word_dataset, generate_word_test)
from fbrl.train import (train_model, check_attention, train_bigram_model,
                         check_bigram_attention, train_word_model, train_motor_model)
from fbrl.evaluate import (test_model, visualize_model, generate_atlas, test_bigram_model,
                           generate_bigram_atlas, test_word_model, generate_word_atlas,
                           test_word_isolation, generate_isolation_atlas)
from fbrl._motor_eval import test_motor_model, generate_motor_atlas
from fbrl.config import load_config


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


def _add_train_overrides(parser):
    """Add common runtime override args to a training subparser."""
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--device', default=None, choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--resume', default=None)
    parser.add_argument('--transfer', default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--save_dir', default=None)
    parser.add_argument('--checkpoint_interval', type=int, default=None)
    parser.add_argument('--guide_weight', type=float, default=None)
    parser.add_argument('--scan_guide_weight', type=float, default=None)
    parser.add_argument('--scaffold_ratio', type=float, default=None)
    parser.add_argument('--scaffold_floor', type=float, default=None)
    parser.add_argument('--scaffold_epochs', type=int, default=None)
    parser.add_argument('--diversity_weight', type=float, default=None)
    parser.add_argument('--diversity_sigma', type=float, default=None)
    parser.add_argument('--scan_vy', type=float, default=None)
    parser.add_argument('--read_vy', type=float, default=None)
    parser.add_argument('--content_weight', type=float, default=None)
    parser.add_argument('--edge_weight', type=float, default=None)
    parser.add_argument('--blur_sigma_ratio', type=float, default=None)
    parser.add_argument('--n_scan_glimpses', type=int, default=None)
    parser.add_argument('--n_read_glimpses', type=int, default=None)
    parser.add_argument('--n_positions', type=int, default=None)
    parser.add_argument('--scan_patch_size', default=None)
    parser.add_argument('--read_patch_size', type=int, default=None)
    parser.add_argument('--n_scales', type=int, default=None)
    # Letter-specific
    parser.add_argument('--recode_weight', type=float, default=None)
    parser.add_argument('--diversity_vy', type=float, default=None)
    # Bigram-specific
    parser.add_argument('--mask_weight', type=float, default=None)
    # Word-specific
    parser.add_argument('--isolation_weight', type=float, default=None)
    parser.add_argument('--isolation_data_dir', default=None)
    parser.add_argument('--isolation_random_prob', type=float, default=None)
    parser.add_argument('--multi_head', action='store_true', default=None)
    parser.add_argument('--amp', action='store_true', default=None)
    # Grouped read
    parser.add_argument('--read_anchor_scan_indices', type=int, nargs='+', default=None)
    parser.add_argument('--n_read_per_group', type=int, default=None)
    # Motor v2 enhanced losses
    parser.add_argument('--case_filter', default=None)
    parser.add_argument('--render_sigma', type=float, default=None)
    parser.add_argument('--latent_match_weight', type=float, default=None)
    parser.add_argument('--frozen_rr_weight', type=float, default=None)
    parser.add_argument('--render_match_weight', type=float, default=None)


def _build_overrides(args):
    """Extract non-None CLI overrides as a dict for config merging."""
    overrides = {}
    # Gather all override fields (skip 'command' and 'config')
    for k, v in vars(args).items():
        if k in ('command', 'config'):
            continue
        if v is not None:
            if k == 'scan_patch_size':
                v = _parse_patch_size(v)
            elif k == 'read_anchor_scan_indices' and isinstance(v, list):
                v = tuple(v)
            overrides[k] = v
    return overrides


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vision-Only Letter Training Pipeline')
    subparsers = parser.add_subparsers(dest='command', required=True)

    gen_parser = subparsers.add_parser('generate')
    gen_parser.add_argument('--letters', default='Aa-Zz')
    gen_parser.add_argument('--num_variants', type=int, default=20)
    gen_parser.add_argument('--noise_level', type=float, default=0.1)
    gen_parser.add_argument('--output_dir', default='data/letters')
    gen_parser.add_argument('--fonts', default='all',
                            help='Font spec: "all", "default", or comma-separated names')

    gentest_parser = subparsers.add_parser('generate_test')
    gentest_parser.add_argument('--letters', default='Aa-Zz')
    gentest_parser.add_argument('--output_dir', default='data/letter_test')
    gentest_parser.add_argument('--fonts', default='all',
                                help='Font spec: "all", "default", or comma-separated names')

    # --- Training subcommands (config-based) ---
    train_parser = subparsers.add_parser('train')
    _add_train_overrides(train_parser)

    train_bi_parser = subparsers.add_parser('train_bigrams')
    _add_train_overrides(train_bi_parser)

    train_w_parser = subparsers.add_parser('train_words')
    _add_train_overrides(train_w_parser)

    # --- Test/eval/atlas subcommands (unchanged) ---
    test_parser = subparsers.add_parser('test')
    test_parser.add_argument('--model_dir', required=True)
    test_parser.add_argument('--test_data_dir', required=True)
    test_parser.add_argument('--output_dir', default='letter_results')
    test_parser.add_argument('--device', default='auto',
                             choices=['auto', 'cpu', 'cuda'])

    viz_parser = subparsers.add_parser('visualize')
    viz_parser.add_argument('--model_dir', required=True)
    viz_parser.add_argument('--data_dir', required=True)
    viz_parser.add_argument('--output_dir', default='letter_visualizations')
    viz_parser.add_argument('--device', default='auto',
                            choices=['auto', 'cpu', 'cuda'])

    atlas_parser = subparsers.add_parser('atlas')
    atlas_parser.add_argument('--model_dir', required=True)
    atlas_parser.add_argument('--test_data_dir', required=True)
    atlas_parser.add_argument('--output', default='data/letter_atlas.html')
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
    chk_parser.add_argument('--diversity_vy', type=float, default=1.0)

    compress_parser = subparsers.add_parser('compress_model')
    compress_parser.add_argument('--input', required=True)
    compress_parser.add_argument('--output', required=True)

    # --- Bigram subcommands (non-training unchanged) ---
    gen_bi_parser = subparsers.add_parser('generate_bigrams')
    gen_bi_parser.add_argument('--num_variants', type=int, default=20)
    gen_bi_parser.add_argument('--noise_level', type=float, default=0.1)
    gen_bi_parser.add_argument('--output_dir', default='data/bigrams')
    gen_bi_parser.add_argument('--fonts', default='default',
                               help='Font spec: "all", "default", or comma-separated names')

    gen_bi_test_parser = subparsers.add_parser('generate_bigrams_test')
    gen_bi_test_parser.add_argument('--output_dir', default='data/bigram_test')
    gen_bi_test_parser.add_argument('--fonts', default='default',
                                    help='Font spec: "all", "default", or comma-separated names')

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
    chk_bi_parser.add_argument('--diversity_vy', type=float, default=1.0)

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

    # --- Word subcommands (non-training unchanged) ---
    gen_w_parser = subparsers.add_parser('generate_words')
    gen_w_parser.add_argument('--num_variants', type=int, default=20)
    gen_w_parser.add_argument('--noise_level', type=float, default=0.1)
    gen_w_parser.add_argument('--output_dir', default='data/words')
    gen_w_parser.add_argument('--fonts', default='default',
                              help='Font spec: "all", "default", or comma-separated names')

    gen_w_test_parser = subparsers.add_parser('generate_words_test')
    gen_w_test_parser.add_argument('--output_dir', default='data/word_test')
    gen_w_test_parser.add_argument('--fonts', default='default',
                                   help='Font spec: "all", "default", or comma-separated names')

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

    iso_atlas_parser = subparsers.add_parser('isolation_atlas')
    iso_atlas_parser.add_argument('--model_dir', required=True)
    iso_atlas_parser.add_argument('--test_data_dir', required=True)
    iso_atlas_parser.add_argument('--output', default='data/isolation_atlas.html')
    iso_atlas_parser.add_argument('--device', default='auto',
                                   choices=['auto', 'cpu', 'cuda'])

    # --- Motor subcommands ---
    gen_traj_parser = subparsers.add_parser('generate_trajectories')
    gen_traj_parser.add_argument('--output_dir', default='data/trajectories')
    gen_traj_parser.add_argument('--font', default='dejavu-sans')
    gen_traj_parser.add_argument('--n_points', type=int, default=32)
    gen_traj_parser.add_argument('--letters', default='Aa-Zz')

    train_motor_parser = subparsers.add_parser('train_motor')
    _add_train_overrides(train_motor_parser)

    test_motor_parser = subparsers.add_parser('test_motor')
    test_motor_parser.add_argument('--model_dir', required=True)
    test_motor_parser.add_argument('--test_data_dir', required=True)
    test_motor_parser.add_argument('--output_dir', default='motor_results')
    test_motor_parser.add_argument('--trajectory_data_dir', default='data/trajectories')
    test_motor_parser.add_argument('--device', default='auto',
                                    choices=['auto', 'cpu', 'cuda'])

    motor_atlas_parser = subparsers.add_parser('motor_atlas')
    motor_atlas_parser.add_argument('--model_dir', required=True)
    motor_atlas_parser.add_argument('--test_data_dir', required=True)
    motor_atlas_parser.add_argument('--output', default='data/motor_atlas.html')
    motor_atlas_parser.add_argument('--trajectory_data_dir', default='data/trajectories')
    motor_atlas_parser.add_argument('--device', default='auto',
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
        cfg = load_config(args.config, _build_overrides(args))
        train_model(cfg)

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
        cfg = load_config(args.config, _build_overrides(args))
        train_bigram_model(cfg)

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
        cfg = load_config(args.config, _build_overrides(args))
        train_word_model(cfg)

    elif args.command == 'test_words':
        test_word_model(args.model_dir, args.test_data_dir, args.output_dir,
                         device=args.device)

    elif args.command == 'test_word_isolation':
        test_word_isolation(args.model_dir, args.test_data_dir, args.output_dir,
                             device=args.device)

    elif args.command == 'word_atlas':
        generate_word_atlas(args.model_dir, args.test_data_dir, args.output,
                             device=args.device)

    elif args.command == 'isolation_atlas':
        generate_isolation_atlas(args.model_dir, args.test_data_dir, args.output,
                                  device=args.device)

    # --- Motor commands ---
    elif args.command == 'generate_trajectories':
        from fbrl.motor import generate_trajectory_dataset
        letters = _parse_letters(args.letters)
        generate_trajectory_dataset(args.output_dir, font_name=args.font,
                                     n_points=args.n_points, letters=letters)

    elif args.command == 'train_motor':
        cfg = load_config(args.config, _build_overrides(args))
        train_motor_model(cfg)

    elif args.command == 'test_motor':
        test_motor_model(args.model_dir, args.test_data_dir, args.output_dir,
                          trajectory_data_dir=args.trajectory_data_dir,
                          device=args.device)

    elif args.command == 'motor_atlas':
        generate_motor_atlas(args.model_dir, args.test_data_dir, args.output,
                              trajectory_data_dir=args.trajectory_data_dir,
                              device=args.device)
