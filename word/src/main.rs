//! CLI entry point for word model training phases.

use fbrl_word::word::{
    load_word_dataset,
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
        _ => {
            eprintln!("Usage: fbrl-word <command> [options]");
            eprintln!();
            eprintln!("Commands:");
            eprintln!("  train-subscan  Step 2: SubScan position denoising");
            eprintln!();
            eprintln!("train-subscan options:");
            eprintln!("  --config <path>         Config JSON");
            eprintln!("  --word-data <path>      Word dataset directory");
            eprintln!("  --checkpoint <path>     Letter model checkpoint (eval only)");
            eprintln!("  --save-dir <path>       Output directory (default: training)");
            eprintln!("  --epochs <n>            Number of epochs");
            eprintln!("  --batch-size <n>        Batch size");
            eprintln!("  --subscan-lr <f>        Learning rate");
            eprintln!("  --max-attempts <n>      Max retries per letter (default: 30)");
            eprintln!("  --fail-penalty <f>      Negative reward on failed read (default: -0.1)");
            eprintln!("  --target-bonus <f>      Extra reward for correct letter (default: 0.2)");
            eprintln!("  --action-sigma <f>      REINFORCE action noise std (default: 0.03)");
            eprintln!("  --threshold <f>         Softmax confidence threshold (default: 0.5)");
            eprintln!("  --monitor <port>        Live monitor port");
            std::process::exit(1);
        }
    }
}

fn run_train_subscan(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut cfg = SubScanTrainConfig::default();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--config" => {
                i += 1;
                let text = std::fs::read_to_string(&args[i])?;
                cfg = serde_json::from_str(&text)?;
            }
            "--word-data" => { i += 1; cfg.word_data = args[i].clone(); }
            "--checkpoint" => { i += 1; cfg.letter_checkpoint = args[i].clone(); }
            "--save-dir" => { i += 1; cfg.save_dir = args[i].clone(); }
            "--epochs" => { i += 1; cfg.epochs = args[i].parse()?; }
            "--batch-size" => { i += 1; cfg.batch_size = args[i].parse()?; }
            "--subscan-lr" => { i += 1; cfg.subscan_lr = args[i].parse()?; }
            "--max-attempts" => { i += 1; cfg.max_attempts = args[i].parse()?; }
            "--fail-penalty" => { i += 1; cfg.fail_penalty = args[i].parse()?; }
            "--target-bonus" => { i += 1; cfg.target_bonus = args[i].parse()?; }
            "--action-sigma" => { i += 1; cfg.action_sigma = args[i].parse()?; }
            "--threshold" => { i += 1; cfg.confidence_threshold = args[i].parse()?; }
            "--monitor" => { i += 1; cfg.monitor_port = Some(args[i].parse()?); }
            other => {
                return Err(format!("unknown option: {other}").into());
            }
        }
        i += 1;
    }

    if cfg.word_data.is_empty() {
        return Err("--word-data is required".into());
    }

    eprintln!("=== SubScan Training (Step 2) — REINFORCE ===");
    eprintln!("Word data:    {}", cfg.word_data);
    eprintln!("Checkpoint:   {}", if cfg.letter_checkpoint.is_empty() { "(none)" } else { &cfg.letter_checkpoint });
    eprintln!("Epochs:       {}", cfg.epochs);
    eprintln!("Batch size:   {}", cfg.batch_size);
    eprintln!("SubScan LR:   {}", cfg.subscan_lr);
    eprintln!("Max attempts: {}", cfg.max_attempts);
    eprintln!("Noise x:      {} → {} over {:.0}% of epochs",
        cfg.noise_x_start, cfg.noise_x_end, cfg.noise_ramp_pct * 100.0);
    eprintln!("Noise y:      {} → {} over {:.0}% of epochs",
        cfg.noise_y_start, cfg.noise_y_end, cfg.noise_ramp_pct * 100.0);
    eprintln!("Fail penalty: {}  Target bonus: {}  σ: {}  Threshold: {}",
        cfg.fail_penalty, cfg.target_bonus, cfg.action_sigma, cfg.confidence_threshold);
    eprintln!("Save dir:     {}", cfg.save_dir);
    eprintln!();

    let word_ds = load_word_dataset(&cfg.word_data)?;

    train_subscan(&cfg, &word_ds, Some(&|s: &SubScanEpochStats| {
        eprintln!(
            "epoch {:3}  attempts={:.1}(max {})  success={:.1}%  target={:.1}%  \
             reward={:.3}  noise=({:.3},{:.3})  lr={:.6}  [{:?}]  ETA {:?}",
            s.epoch + 1,
            s.mean_attempts, s.max_attempt,
            s.success_rate * 100.0,
            s.target_acc * 100.0,
            s.mean_reward,
            s.noise_x, s.noise_y,
            s.lr,
            s.duration, s.eta,
        );
    }))?;

    Ok(())
}
