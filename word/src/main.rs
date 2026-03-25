//! CLI entry point for word model training phases.
//!
//! Subcommands:
//!   train-subscan  — Step 2: SubScan + Letter composition training
//!   train-word     — Step 3: Full word model training (future)
//!   eval           — Evaluate a trained model (future)

use fbrl_word::word::{
    load_word_dataset, load_isolation_dataset,
    SubScanTrainConfig, SubScanEpochStats, train_subscan,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("");

    match cmd {
        "train-subscan" => {
            if let Err(e) = run_train_subscan(&args[2..]) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        "train-word" => {
            eprintln!("Word model training not yet implemented");
            std::process::exit(1);
        }
        "eval" => {
            eprintln!("Evaluation not yet implemented");
            std::process::exit(1);
        }
        _ => {
            eprintln!("Usage: fbrl-word <command> [options]");
            eprintln!();
            eprintln!("Commands:");
            eprintln!("  train-subscan  Step 2: SubScan + Letter composition");
            eprintln!("  train-word     Step 3: Full word model");
            eprintln!("  eval           Evaluate trained model");
            eprintln!();
            eprintln!("train-subscan options:");
            eprintln!("  --config <path>      Config JSON (optional, uses defaults)");
            eprintln!("  --word-data <path>   Word dataset directory");
            eprintln!("  --iso-data <path>    Isolation letter dataset directory");
            eprintln!("  --checkpoint <path>  Letter model checkpoint");
            eprintln!("  --save-dir <path>    Output directory (default: training)");
            eprintln!("  --epochs <n>         Number of epochs");
            eprintln!("  --monitor <port>     Live monitor port");
            std::process::exit(1);
        }
    }
}

fn run_train_subscan(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut cfg = SubScanTrainConfig::default();

    // Parse CLI args.
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--config" => {
                i += 1;
                let text = std::fs::read_to_string(&args[i])?;
                cfg = serde_json::from_str(&text)?;
            }
            "--word-data" => { i += 1; cfg.word_data = args[i].clone(); }
            "--iso-data" => { i += 1; cfg.isolation_data = args[i].clone(); }
            "--checkpoint" => { i += 1; cfg.letter_checkpoint = args[i].clone(); }
            "--save-dir" => { i += 1; cfg.save_dir = args[i].clone(); }
            "--epochs" => { i += 1; cfg.epochs = args[i].parse()?; }
            "--monitor" => { i += 1; cfg.monitor_port = Some(args[i].parse()?); }
            "--batch-size" => { i += 1; cfg.batch_size = args[i].parse()?; }
            "--subscan-lr" => { i += 1; cfg.subscan_lr = args[i].parse()?; }
            "--scan-lr" => { i += 1; cfg.scan_lr = args[i].parse()?; }
            other => {
                return Err(format!("unknown option: {other}").into());
            }
        }
        i += 1;
    }

    if cfg.word_data.is_empty() {
        return Err("--word-data is required".into());
    }
    if cfg.isolation_data.is_empty() {
        return Err("--iso-data is required".into());
    }

    eprintln!("=== SubScan Training (Step 2) ===");
    eprintln!("Word data:    {}", cfg.word_data);
    eprintln!("Iso data:     {}", cfg.isolation_data);
    eprintln!("Checkpoint:   {}", if cfg.letter_checkpoint.is_empty() { "(none)" } else { &cfg.letter_checkpoint });
    eprintln!("Epochs:       {}", cfg.epochs);
    eprintln!("Batch size:   {}", cfg.batch_size);
    eprintln!("SubScan LR:   {}", cfg.subscan_lr);
    eprintln!("Scan LR:      {}", cfg.scan_lr);
    eprintln!("Save dir:     {}", cfg.save_dir);
    eprintln!();

    let word_ds = load_word_dataset(&cfg.word_data)?;
    let iso_ds = load_isolation_dataset(&cfg.isolation_data)?;

    train_subscan(&cfg, &word_ds, &iso_ds, Some(&|s: &SubScanEpochStats| {
        eprintln!(
            "epoch {:3}  ce={:.4}  recon={:.4}  div={:.4}  total={:.4}  \
             acc={:.1}%  noise={:.3}  [{:?}]  ETA {:?}",
            s.epoch + 1,
            s.ce_loss, s.recon_loss, s.div_loss, s.total_loss,
            s.accuracy * 100.0, s.noise_range,
            s.duration, s.eta,
        );
    }))?;

    Ok(())
}
