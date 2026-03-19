//! CLI entry point for word model training and evaluation.

use fbrl_word::word::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    run_train(&args);
}

fn run_train(args: &[String]) {
    let mut cfg = WordConfig::default();
    let mut data_dir = String::new();
    let mut synthetic = 0usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            // Data source.
            "--data" => { data_dir = next_arg(args, &mut i); }
            "--synthetic" => { synthetic = next_arg(args, &mut i).parse().expect("--synthetic N"); }

            // Output.
            "--save" => { cfg.save_dir = next_arg(args, &mut i); }

            // Architecture.
            "--positions" => { cfg.n_positions = next_arg(args, &mut i).parse().expect("--positions N"); }
            "--scan" => { cfg.n_scan = next_arg(args, &mut i).parse().expect("--scan N"); }
            "--read" => { cfg.n_read = next_arg(args, &mut i).parse().expect("--read N"); }
            "--patch-size" => { cfg.patch_size = next_arg(args, &mut i).parse().expect("--patch-size N"); }
            "--scan-patch-w" => { cfg.scan_patch_w = next_arg(args, &mut i).parse().expect("--scan-patch-w N"); }
            "--latent-dim" => { cfg.latent_dim = next_arg(args, &mut i).parse().expect("--latent-dim N"); }

            // Training.
            "--epochs" => { cfg.epochs = next_arg(args, &mut i).parse().expect("--epochs N"); }
            "--batch-size" => { cfg.batch_size = next_arg(args, &mut i).parse().expect("--batch-size N"); }
            "--lr" => { cfg.lr = next_arg(args, &mut i).parse().expect("--lr F"); }
            "--min-lr" => { cfg.min_lr = next_arg(args, &mut i).parse().expect("--min-lr F"); }
            "--max-grad-norm" => { cfg.max_grad_norm = next_arg(args, &mut i).parse().expect("--max-grad-norm F"); }

            // Loss weights.
            "--scan-guide-weight" => { cfg.scan_guide_weight = next_arg(args, &mut i).parse().expect("F"); }
            "--read-guide-weight" => { cfg.read_guide_weight = next_arg(args, &mut i).parse().expect("F"); }
            "--diversity-weight" => { cfg.diversity_weight = next_arg(args, &mut i).parse().expect("F"); }
            "--recon-weight" => { cfg.recon_weight = next_arg(args, &mut i).parse().expect("F"); }
            "--content-weight" => { cfg.content_weight = next_arg(args, &mut i).parse().expect("F"); }
            "--isolation-weight" => { cfg.isolation_weight = next_arg(args, &mut i).parse().expect("F"); }
            "--scan-vy" => { cfg.scan_vy = next_arg(args, &mut i).parse().expect("F"); }
            "--read-vy" => { cfg.read_vy = next_arg(args, &mut i).parse().expect("F"); }

            // Scaffold.
            "--scaffold-ratio" => { cfg.scaffold_ratio = next_arg(args, &mut i).parse().expect("F"); }
            "--scaffold-floor" => { cfg.scaffold_floor = next_arg(args, &mut i).parse().expect("F"); }

            // Data.
            "--isolation-data" => { cfg.isolation_data_dir = next_arg(args, &mut i); }

            // Transfer.
            "--transfer" => { cfg.transfer_from = next_arg(args, &mut i); }

            // Checkpointing & monitoring.
            "--checkpoint-interval" => { cfg.checkpoint_interval = next_arg(args, &mut i).parse().expect("N"); }
            "--monitor" => { cfg.monitor_port = Some(next_arg(args, &mut i).parse().expect("PORT")); }

            "--help" | "-h" => { usage(); std::process::exit(0); }
            other => {
                eprintln!("unknown flag: {other}");
                usage();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // Load or generate dataset.
    let ds = if synthetic > 0 {
        let samples = new_synthetic_dataset(synthetic).expect("synthetic dataset");
        eprintln!("Generated {} synthetic word samples", samples.len());
        WordDataset { samples }
    } else if !data_dir.is_empty() {
        load_word_dataset(&data_dir).expect("load word dataset")
    } else {
        eprintln!("Usage: fbrl-word --data <dir> or --synthetic <N>");
        usage();
        std::process::exit(1);
    };

    // Load isolation dataset if specified.
    let iso_ds = if !cfg.isolation_data_dir.is_empty() {
        Some(load_isolation_dataset(&cfg.isolation_data_dir).expect("load isolation data"))
    } else {
        None
    };

    let scaffold_epochs = (cfg.scaffold_ratio * cfg.epochs as f64).round() as usize;

    eprintln!(
        "Training: {} epochs, batch {}, lr {:.4}\u{2192}{:.4}, {} scan + {} read, {} positions",
        cfg.epochs, cfg.batch_size, cfg.lr, cfg.min_lr,
        cfg.n_scan, cfg.n_read, cfg.n_positions,
    );
    eprintln!(
        "Scaffold: {} epochs ({:.0}%), guide={:.1}/{:.1}, diversity={:.1}, content={:.1}",
        scaffold_epochs, cfg.scaffold_ratio * 100.0,
        cfg.scan_guide_weight, cfg.read_guide_weight,
        cfg.diversity_weight, cfg.content_weight,
    );
    if !cfg.transfer_from.is_empty() {
        eprintln!("Transfer: {}", cfg.transfer_from);
    }
    if !cfg.save_dir.is_empty() {
        eprintln!("Output: {}", cfg.save_dir);
    }
    if let Some(port) = cfg.monitor_port {
        eprintln!("Monitor: http://localhost:{port}");
    }
    eprintln!();

    train_word(&cfg, &ds, iso_ds.as_ref(), Some(&|s: &WordEpochStats| {
        let pos_str = (0..4).map(|p| format!(
            "P{}={:.4}({:.0}%)", p + 1,
            s.position_losses[p], s.position_accs[p] * 100.0
        )).collect::<Vec<_>>().join("  ");
        eprintln!(
            "Epoch {:3}  {}  Recon {:.4}  Guide {:.4}  Div {:.4}  Cont {:.4}  \
             Hit {:.0}%  scaff {:.2}  lr {:.6}  [{:?}  ETA {:?}]",
            s.epoch + 1, pos_str,
            s.recon_loss, s.guide_loss, s.div_loss, s.content_loss,
            s.hit_rate * 100.0, s.scaffold_weight, s.lr,
            s.duration, s.eta,
        );
    }))
    .expect("training failed");

    eprintln!("\nTraining complete.");
    if !cfg.save_dir.is_empty() {
        eprintln!("Model:     {}/model_final.fdl.gz", cfg.save_dir);
        eprintln!("Manifest:  {}/manifest.json", cfg.save_dir);
        eprintln!("Plot:      {}/training.html", cfg.save_dir);
        eprintln!("CSV:       {}/training.csv", cfg.save_dir);
        eprintln!("Log:       {}/training.log", cfg.save_dir);
        if cfg.monitor_port.is_some() {
            eprintln!("Dashboard: {}/dashboard.html", cfg.save_dir);
        }
    }
}

fn next_arg(args: &[String], i: &mut usize) -> String {
    *i += 1;
    args.get(*i)
        .unwrap_or_else(|| {
            eprintln!("missing value for {}", args[*i - 1]);
            std::process::exit(1);
        })
        .clone()
}

fn usage() {
    eprintln!("fbrl-word -- foveal attention word recognition");
    eprintln!();
    eprintln!("Training:");
    eprintln!("  fbrl-word --data <dir> [--save <dir>] [--monitor <port>] [options]");
    eprintln!("  fbrl-word --synthetic <N> [--epochs <N>]");
    eprintln!();
    eprintln!("Data:");
    eprintln!("  --data <dir>             word training data directory");
    eprintln!("  --synthetic <N>          generate N synthetic samples");
    eprintln!("  --isolation-data <dir>   single-letter images for isolation loss");
    eprintln!("  --save <dir>             output directory (default: training)");
    eprintln!();
    eprintln!("Architecture:");
    eprintln!("  --positions <N>          letter positions (default: 4)");
    eprintln!("  --scan <N>               scan glimpses (default: 8)");
    eprintln!("  --read <N>               read glimpses (default: 12)");
    eprintln!("  --patch-size <N>         glimpse patch size (default: 12)");
    eprintln!("  --scan-patch-w <N>       scan patch width (default: 18)");
    eprintln!("  --latent-dim <N>         hidden dimension (default: 256)");
    eprintln!();
    eprintln!("Training:");
    eprintln!("  --epochs <N>             training epochs (default: 200)");
    eprintln!("  --batch-size <N>         batch size (default: 32)");
    eprintln!("  --lr <F>                 initial learning rate (default: 0.001)");
    eprintln!("  --transfer <path>        letter model checkpoint for transfer");
    eprintln!("  --scaffold-ratio <F>     scaffold duration ratio (default: 0.67)");
    eprintln!();
    eprintln!("Other:");
    eprintln!("  --checkpoint-interval <N> save every N epochs (default: 50)");
    eprintln!("  --monitor <PORT>         live dashboard on http://localhost:PORT");
}
