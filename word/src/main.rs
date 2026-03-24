//! CLI entry point for word model training phases.
//!
//! Subcommands:
//!   train-subscan  — Step 2: SubScan + Letter composition training
//!   train-word     — Step 3: Full word model training (future)
//!   eval           — Evaluate a trained model (future)

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("");

    match cmd {
        "train-subscan" => {
            eprintln!("SubScan training not yet implemented");
            std::process::exit(1);
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
            eprintln!("Usage: fbrl-word <command>");
            eprintln!();
            eprintln!("Commands:");
            eprintln!("  train-subscan  Step 2: SubScan + Letter composition");
            eprintln!("  train-word     Step 3: Full word model");
            eprintln!("  eval           Evaluate trained model");
            std::process::exit(1);
        }
    }
}
