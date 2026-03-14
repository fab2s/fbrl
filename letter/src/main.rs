//! CLI entry point for letter model training.

use fbrl::letter::*;

fn main() {
    let mut cfg = LetterConfig::default();
    let mut data_dir = String::new();
    let mut synthetic = 0usize;

    // Parse args: --key value pairs.
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            // Data source.
            "--data" => { data_dir = next_arg(&args, &mut i); }
            "--synthetic" => { synthetic = next_arg(&args, &mut i).parse().expect("--synthetic N"); }

            // Output.
            "--save" => { cfg.save_dir = next_arg(&args, &mut i); }

            // Architecture.
            "--scan" => { cfg.n_scan = next_arg(&args, &mut i).parse().expect("--scan N"); }
            "--read" => { cfg.n_read = next_arg(&args, &mut i).parse().expect("--read N"); }
            "--patch-size" => { cfg.patch_size = next_arg(&args, &mut i).parse().expect("--patch-size N"); }
            "--scan-patch-w" => { cfg.scan_patch_w = next_arg(&args, &mut i).parse().expect("--scan-patch-w N"); }
            "--scales" => { cfg.n_scales = next_arg(&args, &mut i).parse().expect("--scales N"); }
            "--latent-dim" => { cfg.latent_dim = next_arg(&args, &mut i).parse().expect("--latent-dim N"); }

            // Training.
            "--epochs" => { cfg.epochs = next_arg(&args, &mut i).parse().expect("--epochs N"); }
            "--batch-size" => { cfg.batch_size = next_arg(&args, &mut i).parse().expect("--batch-size N"); }
            "--lr" => { cfg.lr = next_arg(&args, &mut i).parse().expect("--lr F"); }
            "--min-lr" => { cfg.min_lr = next_arg(&args, &mut i).parse().expect("--min-lr F"); }
            "--max-grad-norm" => { cfg.max_grad_norm = next_arg(&args, &mut i).parse().expect("--max-grad-norm F"); }

            // Loss weights.
            "--scan-guide-weight" => { cfg.scan_guide_weight = next_arg(&args, &mut i).parse().expect("--scan-guide-weight F"); }
            "--read-guide-weight" => { cfg.read_guide_weight = next_arg(&args, &mut i).parse().expect("--read-guide-weight F"); }
            "--diversity-weight" => { cfg.diversity_weight = next_arg(&args, &mut i).parse().expect("--diversity-weight F"); }
            "--recon-weight" => { cfg.recon_weight = next_arg(&args, &mut i).parse().expect("--recon-weight F"); }

            // Checkpointing & monitoring.
            "--checkpoint-interval" => { cfg.checkpoint_interval = next_arg(&args, &mut i).parse().expect("--checkpoint-interval N"); }
            "--monitor" => { cfg.monitor_port = Some(next_arg(&args, &mut i).parse().expect("--monitor PORT")); }

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
        let ds = new_synthetic_dataset(synthetic).expect("synthetic dataset");
        eprintln!("Generated {} synthetic samples", ds.len());
        ds
    } else if !data_dir.is_empty() {
        load_letter_dataset(&data_dir).expect("load dataset")
    } else {
        eprintln!("Usage: fbrl --data <dir> or --synthetic <N>");
        usage();
        std::process::exit(1);
    };

    // Train.
    eprintln!(
        "Training: {} epochs, batch {}, lr {:.4}\u{2192}{:.4} (cosine), {} scan + {} read",
        cfg.epochs, cfg.batch_size, cfg.lr, cfg.min_lr, cfg.n_scan, cfg.n_read,
    );
    if !cfg.save_dir.is_empty() {
        eprintln!("Output: {}", cfg.save_dir);
    }
    if let Some(port) = cfg.monitor_port {
        eprintln!("Monitor: http://localhost:{port}");
    }
    eprintln!();

    train_letter(&cfg, &ds, Some(&|s: &EpochStats| {
        eprintln!(
            "Epoch {:3}  Ltr {:.4}({:.0}%)  Case {:.4}({:.0}%)  Recon {:.4}  \
             Guide {:.4}  Div {:.4}  Hit {:.0}%  lr {:.6}  [{:?}  ETA {:?}]",
            s.epoch + 1,
            s.letter_loss, s.letter_acc * 100.0,
            s.case_loss, s.case_acc * 100.0,
            s.recon_loss, s.guide_loss, s.div_loss,
            s.hit_rate * 100.0, s.lr,
            s.duration, s.eta,
        );
    }))
    .expect("training failed");

    eprintln!("\nTraining complete.");
    if !cfg.save_dir.is_empty() {
        eprintln!("Model:     {}/model_final.fdl", cfg.save_dir);
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
    eprintln!("fbrl -- foveal attention letter recognition trainer");
    eprintln!();
    eprintln!("Data:");
    eprintln!("  --data <dir>             training data directory");
    eprintln!("  --synthetic <N>          generate N synthetic samples");
    eprintln!("  --save <dir>             output directory (default: training)");
    eprintln!();
    eprintln!("Architecture:");
    eprintln!("  --scan <N>               scan glimpses (default: 1)");
    eprintln!("  --read <N>               read glimpses (default: 6)");
    eprintln!("  --patch-size <N>         glimpse patch size (default: 12)");
    eprintln!("  --scan-patch-w <N>       scan patch width (default: 18)");
    eprintln!("  --scales <N>             multi-resolution scales (default: 1)");
    eprintln!("  --latent-dim <N>         hidden dimension (default: 128)");
    eprintln!();
    eprintln!("Training:");
    eprintln!("  --epochs <N>             training epochs (default: 100)");
    eprintln!("  --batch-size <N>         batch size (default: 32)");
    eprintln!("  --lr <F>                 initial learning rate (default: 0.001)");
    eprintln!("  --min-lr <F>             cosine floor (default: 0.0)");
    eprintln!("  --max-grad-norm <F>      gradient clipping (default: 5.0)");
    eprintln!();
    eprintln!("Loss weights:");
    eprintln!("  --scan-guide-weight <F>  scan attention guide (default: 8.0)");
    eprintln!("  --read-guide-weight <F>  read attention guide (default: 0.0)");
    eprintln!("  --diversity-weight <F>   fixation diversity (default: 1.0)");
    eprintln!("  --recon-weight <F>       reconstruction (default: 1.0)");
    eprintln!();
    eprintln!("Other:");
    eprintln!("  --checkpoint-interval <N> save every N epochs (default: 50)");
    eprintln!("  --monitor <PORT>         live dashboard on http://localhost:PORT");
}
