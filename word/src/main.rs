//! CLI entry point for word model training phases.

use fbrl_word::word::{
    load_word_dataset,
    SubScanTrainConfig, SubScanEpochStats, train_subscan,
    SubScanEvalConfig, eval_subscan_composition,
    GenConfig, generate_word_dataset, generate_letter_dataset, save_dataset,
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
        "eval-subscan" => {
            if let Err(e) = run_eval_subscan(&args[2..]) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        "generate" => {
            if let Err(e) = run_generate(&args[2..]) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        _ => {
            eprintln!("Usage: fbrl-word <command> [options]");
            eprintln!();
            eprintln!("Commands:");
            eprintln!("  train-subscan  Step 2: triangle SubScan centering");
            eprintln!("  eval-subscan   Composition eval: SubScan + LetterModel");
            eprintln!("  generate       Generate dataset from fonts");
            eprintln!();
            eprintln!("train-subscan options:");
            eprintln!("  --word-data <path>      Word dataset directory");
            eprintln!("  --save-dir <path>       Output directory");
            eprintln!("  --epochs <n>            Number of epochs");
            eprintln!("  --batch-size <n>        Batch size");
            eprintln!("  --subscan-lr <f>        Learning rate");
            eprintln!("  --monitor <port>        Live monitor port");
            eprintln!();
            eprintln!("eval-subscan options:");
            eprintln!("  --word-data <path>      Word dataset directory");
            eprintln!("  --subscan <path>        SubScan checkpoint");
            eprintln!("  --letter <path>         Letter model checkpoint");
            eprintln!("  --noise-x <f>           Input noise x (default: 0.10)");
            eprintln!("  --noise-y <f>           Input noise y (default: 0.05)");
            eprintln!("  --save-dir <path>       Output directory for eval.json");
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
            "--save-dir" => { i += 1; cfg.save_dir = args[i].clone(); }
            "--epochs" => { i += 1; cfg.epochs = args[i].parse()?; }
            "--batch-size" => { i += 1; cfg.batch_size = args[i].parse()?; }
            "--subscan-lr" => { i += 1; cfg.subscan_lr = args[i].parse()?; }
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

    eprintln!("=== SubScan Training (Step 2) — Triangle MSE ===");
    eprintln!("Word data:    {}", cfg.word_data);
    eprintln!("Epochs:       {}", cfg.epochs);
    eprintln!("Batch size:   {}", cfg.batch_size);
    eprintln!("SubScan LR:   {}", cfg.subscan_lr);
    eprintln!("Glimpses:     {}", cfg.subscan_n_glimpses);
    eprintln!("Triangle:     base_hw=[{:.3}, {:.3}]  height={:.3}",
        cfg.min_base_hw, cfg.max_base_hw, cfg.triangle_height);
    eprintln!("Noise x:      {} → {} over {:.0}% of epochs",
        cfg.noise_x_start, cfg.noise_x_end, cfg.noise_ramp_pct * 100.0);
    eprintln!("Noise y:      {} → {} over {:.0}% of epochs",
        cfg.noise_y_start, cfg.noise_y_end, cfg.noise_ramp_pct * 100.0);
    eprintln!("Save dir:     {}", cfg.save_dir);
    eprintln!();

    let word_ds = load_word_dataset(&cfg.word_data)?;

    train_subscan(&cfg, &word_ds, Some(&|s: &SubScanEpochStats| {
        eprintln!(
            "epoch {:3}  mse={:.6}  err_x={:.4}  noise=({:.3},{:.3})  lr={:.6}  [{:?}]  ETA {:?}",
            s.epoch + 1,
            s.mse, s.mean_err_x,
            s.noise_x, s.noise_y,
            s.lr,
            s.duration, s.eta,
        );
    }))?;

    Ok(())
}

fn run_eval_subscan(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut cfg = SubScanEvalConfig::default();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--word-data" => { i += 1; cfg.word_data = args[i].clone(); }
            "--subscan" => { i += 1; cfg.subscan_checkpoint = args[i].clone(); }
            "--letter" => { i += 1; cfg.letter_checkpoint = args[i].clone(); }
            "--noise-x" => { i += 1; cfg.noise_x = args[i].parse()?; }
            "--noise-y" => { i += 1; cfg.noise_y = args[i].parse()?; }
            "--save-dir" => { i += 1; cfg.save_dir = args[i].clone(); }
            "--batch-size" => { i += 1; cfg.batch_size = args[i].parse()?; }
            other => {
                return Err(format!("unknown option: {other}").into());
            }
        }
        i += 1;
    }

    if cfg.word_data.is_empty() {
        return Err("--word-data is required".into());
    }
    if cfg.subscan_checkpoint.is_empty() {
        return Err("--subscan is required".into());
    }
    if cfg.letter_checkpoint.is_empty() {
        return Err("--letter is required".into());
    }

    eprintln!("=== Composition Eval: SubScan + LetterModel ===");
    eprintln!("Word data:  {}", cfg.word_data);
    eprintln!("SubScan:    {}", cfg.subscan_checkpoint);
    eprintln!("Letter:     {}", cfg.letter_checkpoint);
    eprintln!("Noise:      x={:.3}  y={:.3}", cfg.noise_x, cfg.noise_y);
    eprintln!();

    let word_ds = load_word_dataset(&cfg.word_data)?;
    let results = eval_subscan_composition(&cfg, &word_ds)?;

    eprintln!();
    eprintln!("=== Results ===");
    for p in &results.positions {
        let acc = if p.total > 0 { p.correct as f64 / p.total as f64 * 100.0 } else { 0.0 };
        eprintln!("  pos {} (x={:+.2}): {}/{} = {:.1}%  err_x={:.4}",
            p.position, p.gt_center, p.correct, p.total, acc, p.mean_err_x);
    }
    eprintln!();
    eprintln!("  TOTAL: {}/{} = {:.1}%  mean_err_x={:.4}",
        results.correct, results.total, results.accuracy * 100.0, results.mean_err_x);

    Ok(())
}

fn run_generate(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut config_path = String::new();
    let mut save_dir = String::new();
    let mut mode = "word".to_string();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--config" => { i += 1; config_path = args[i].clone(); }
            "--save" => { i += 1; save_dir = args[i].clone(); }
            "--mode" => { i += 1; mode = args[i].clone(); }
            other => {
                return Err(format!("unknown option: {other}").into());
            }
        }
        i += 1;
    }

    if config_path.is_empty() {
        return Err("--config <path.json> is required".into());
    }

    let text = std::fs::read_to_string(&config_path)?;
    let cfg: GenConfig = serde_json::from_str(&text)?;

    eprintln!("=== Generate Dataset ===");
    eprintln!("Config:     {}", config_path);
    eprintln!("Fonts:      {}", cfg.fonts);
    eprintln!("Charset:    {} chars", cfg.charset.len());
    eprintln!("Letters:    {}-{} per image", cfg.min_letters, cfg.max_letters);
    eprintln!("Samples:    {}", cfg.samples);
    eprintln!("Image:      {}×{}", cfg.image_height, cfg.image_width);
    eprintln!();

    match mode.as_str() {
        "letter" => {
            let gds = generate_letter_dataset(&cfg)?;
            eprintln!("Dataset: {} letter samples in memory", gds.dataset.samples.len());
            if !save_dir.is_empty() {
                // Save letter dataset as PNGs + metadata.
                let dir_path = std::path::Path::new(&save_dir);
                std::fs::create_dir_all(dir_path)?;
                let mut meta = std::collections::HashMap::<String, serde_json::Value>::new();
                for (i, (sample, pixels)) in gds.dataset.samples.iter()
                    .zip(gds.pixel_buffers.iter()).enumerate()
                {
                    let img_name = format!("img_{:05}.png", i);
                    let img_path = dir_path.join(&img_name);
                    fbrl_word::word::save_letter_png(
                        pixels, gds.image_height, gds.image_width, &img_path,
                    )?;
                    let letter = ((sample.letter_idx as u8 + b'A') as char).to_string();
                    let case = if sample.case_label > 0.5 { "lower" } else { "upper" };
                    meta.insert(img_name.clone(), serde_json::json!({
                        "image": img_name, "clean": img_name,
                        "letter": letter, "case": case,
                    }));
                }
                std::fs::write(
                    dir_path.join("metadata.json"),
                    serde_json::to_string_pretty(&meta)?,
                )?;
                eprintln!("Saved to {}", save_dir);
            }
        }
        _ => {
            let gds = generate_word_dataset(&cfg)?;
            eprintln!("Dataset: {} word samples in memory", gds.dataset.samples.len());
            if !save_dir.is_empty() {
                save_dataset(&gds, &save_dir)?;
                eprintln!("Saved to {}", save_dir);
            }
        }
    }

    if save_dir.is_empty() {
        eprintln!("(use --save <dir> to write to disk)");
    }

    Ok(())
}
