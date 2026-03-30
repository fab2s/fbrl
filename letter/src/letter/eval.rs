//! Evaluation module — loads a trained model, runs inference on test data,
//! generates accuracy reports, and produces an attention atlas HTML.

use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::Path;

use flodl::autograd::{Variable, NoGradGuard};
use flodl::tensor::{cuda_available, Device, Result, Tensor, TensorError, TensorOptions};
use serde::Deserialize;

use super::data::{load_gray_png, LetterDataset};
use super::model::LetterModel;
use super::train::LetterConfig;

/// Where test data comes from.
pub enum TestSource<'a> {
    /// Load PNGs from a directory with metadata.json.
    Directory(&'a str),
    /// Use an in-memory generated dataset directly.
    Dataset(LetterDataset),
}

/// Test dataset metadata entry.
#[derive(Deserialize)]
struct TestMetaEntry {
    image: String,
    letter: String,
    case: String,
    #[serde(default)]
    font: String,
}

/// A single test sample with metadata.
pub struct TestSample {
    pub image: Tensor,       // [1, H, W]
    pub letter: String,      // "A", "b", etc.
    pub letter_idx: i64,     // 0-25
    pub case_label: f32,     // 0.0=upper, 1.0=lower
    pub case_str: String,    // "upper" or "lower"
    pub font: String,
    pub png_bytes: Vec<u8>,  // original PNG for atlas embedding
}

/// Per-sample evaluation result.
struct SampleResult {
    letter: String,
    letter_idx: i64,
    pred_letter_idx: i64,
    case_str: String,
    case_correct: bool,
    font: String,
    recon_mse: f64,
    scan_locs: Vec<[f64; 2]>,  // (x, y) in [-1, 1]
    read_locs: Vec<[f64; 2]>,
    image_png: Vec<u8>,         // original PNG bytes for atlas
    recon_data: Vec<f32>,       // [H, W] reconstruction pixels
    latent: Vec<f32>,           // [latent_dim] hidden representation
}

/// Load test dataset from a directory with metadata.json.
fn load_test_dataset(dir: &str) -> Result<Vec<TestSample>> {
    let dir_path = Path::new(dir);
    let meta_path = dir_path.join("metadata.json");
    let meta_str = fs::read_to_string(&meta_path)
        .map_err(|e| TensorError::new(&format!("read test metadata: {e}")))?;
    let meta: HashMap<String, TestMetaEntry> = serde_json::from_str(&meta_str)
        .map_err(|e| TensorError::new(&format!("parse test metadata: {e}")))?;

    let mut samples = Vec::with_capacity(meta.len());
    for entry in meta.values() {
        let img_path = resolve_test_path(dir_path, &entry.image);
        let img_tensor = load_gray_png(&img_path)?;
        let png_bytes = fs::read(&img_path).unwrap_or_default();

        let letter_upper = entry.letter.to_uppercase();
        let ch = match letter_upper.as_bytes().first() {
            Some(&b) if b.is_ascii_uppercase() => b,
            _ => continue,
        };
        let letter_idx = (ch - b'A') as i64;
        let case_label = if entry.case == "lower" { 1.0f32 } else { 0.0f32 };

        samples.push(TestSample {
            image: img_tensor,
            letter: entry.letter.clone(),
            letter_idx,
            case_label,
            case_str: entry.case.clone(),
            font: if entry.font.is_empty() { "unknown".into() } else { entry.font.clone() },
            png_bytes,
        });
    }

    // Sort by letter then case then font for consistent output.
    samples.sort_by(|a, b| {
        a.letter_idx.cmp(&b.letter_idx)
            .then(a.case_label.partial_cmp(&b.case_label).unwrap())
            .then(a.font.cmp(&b.font))
    });

    eprintln!("Loaded {} test samples from {}", samples.len(), dir);
    Ok(samples)
}

/// Resolve image path relative to test data directory.
fn resolve_test_path(dir: &Path, path: &str) -> std::path::PathBuf {
    let p = Path::new(path);
    // Try dir-relative first
    let joined = dir.join(p);
    if joined.exists() { return joined; }
    // Try basename only
    if let Some(name) = p.file_name() {
        let by_name = dir.join(name);
        if by_name.exists() { return by_name; }
    }
    joined
}

/// Convert an in-memory LetterDataset into TestSamples.
fn samples_from_dataset(ds: &LetterDataset) -> Result<Vec<TestSample>> {
    let mut samples = Vec::with_capacity(ds.len());
    for s in &ds.samples {
        let shape = s.image.shape(); // [1, H, W]
        let h = shape[1] as usize;
        let w = shape[2] as usize;
        let pixels = s.image.to_f32_vec()?;
        let png_bytes = pixels_to_png(&pixels, w, h);

        let ch = (s.letter_idx as u8 + b'A') as char;
        let letter = if s.case_label > 0.5 {
            ch.to_lowercase().to_string()
        } else {
            ch.to_string()
        };
        let case_str = if s.case_label > 0.5 { "lower" } else { "upper" }.to_string();

        samples.push(TestSample {
            image: s.image.clone(),
            letter,
            letter_idx: s.letter_idx,
            case_label: s.case_label,
            case_str,
            font: if s.font.is_empty() { "unknown".into() } else { s.font.clone() },
            png_bytes,
        });
    }

    samples.sort_by(|a, b| {
        a.letter_idx.cmp(&b.letter_idx)
            .then(a.case_label.partial_cmp(&b.case_label).unwrap())
            .then(a.font.cmp(&b.font))
    });

    eprintln!("Generated {} test samples in memory", samples.len());
    Ok(samples)
}

const LETTERS: &[char] = &[
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
];

fn idx_to_letter(idx: i64, case_str: &str) -> String {
    let ch = LETTERS.get(idx as usize).copied().unwrap_or('?');
    if case_str == "lower" {
        ch.to_lowercase().to_string()
    } else {
        ch.to_string()
    }
}

/// Extract fixation locations from Variable traces as Vec<[x, y]>.
fn extract_locs(traces: &[Variable]) -> Result<Vec<[f64; 2]>> {
    let mut locs = Vec::with_capacity(traces.len());
    for t in traces {
        let data = t.data().to_f32_vec()?;
        // Each trace is [B, 2] — take first batch element
        if data.len() >= 2 {
            locs.push([data[0] as f64, data[1] as f64]);
        }
    }
    Ok(locs)
}

/// Run evaluation: load model, run inference, generate reports.
pub fn eval_letter(
    run_dir: &str,
    test_source: TestSource,
    save_dir: Option<&str>,
) -> Result<()> {
    let run_path = Path::new(run_dir);
    let save_path = save_dir
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("{}/eval", run_dir));

    // Load config from manifest.
    let manifest_path = run_path.join("manifest.json");
    let manifest_str = fs::read_to_string(&manifest_path)
        .map_err(|e| TensorError::new(&format!("read manifest: {e}")))?;
    let manifest: serde_json::Value = serde_json::from_str(&manifest_str)
        .map_err(|e| TensorError::new(&format!("parse manifest: {e}")))?;
    let cfg: LetterConfig = serde_json::from_value(manifest["config"].clone())
        .map_err(|e| TensorError::new(&format!("parse config from manifest: {e}")))?;

    // Find model file.
    let model_file = manifest["files"]["model"]
        .as_str()
        .unwrap_or("model_final.fdl.gz");
    let model_path = run_path.join(model_file);

    let img_h = cfg.img_h;
    let img_w = cfg.img_w;
    eprintln!("Config: {} scan + {} read, latent_dim={}, img={}×{}", cfg.n_scan, cfg.n_read, cfg.latent_dim, img_h, img_w);
    eprintln!("Model:  {}", model_path.display());

    // Create model and load checkpoint.
    let model = LetterModel::new(
        cfg.n_classes, cfg.n_scan, cfg.n_read, cfg.patch_size, cfg.scan_patch_w,
        cfg.n_scales, cfg.latent_dim, cfg.img_h, cfg.img_w,
    )?;

    let device = if cuda_available() {
        eprintln!("Using CUDA");
        model.graph.set_device(Device::CUDA(0));
        Device::CUDA(0)
    } else {
        eprintln!("Using CPU");
        Device::CPU
    };

    let report = model.graph.load_checkpoint(&model_path.to_string_lossy())?;
    eprintln!("Checkpoint: {} loaded, {} skipped, {} missing",
        report.loaded.len(), report.skipped.len(), report.missing.len());
    model.eval();

    // Load test data.
    let samples = match test_source {
        TestSource::Directory(dir) => load_test_dataset(dir)?,
        TestSource::Dataset(ds) => samples_from_dataset(&ds)?,
    };

    // Run inference under NoGradGuard.
    let _guard = NoGradGuard::new();
    let mut results: Vec<SampleResult> = Vec::with_capacity(samples.len());

    for (i, sample) in samples.iter().enumerate() {
        // Pad input image to model's expected dimensions if needed.
        let raw_shape = sample.image.shape(); // [1, H, W]
        let (raw_h, raw_w) = (raw_shape[1], raw_shape[2]);
        let img = if raw_h != img_h || raw_w != img_w {
            let src = sample.image.to_f32_vec()?;
            let mut dst = vec![0.0f32; (img_h * img_w) as usize];
            let off_y = ((img_h - raw_h) / 2).max(0) as usize;
            let off_x = ((img_w - raw_w) / 2).max(0) as usize;
            for y in 0..raw_h as usize {
                for x in 0..raw_w as usize {
                    let dy = y + off_y;
                    let dx = x + off_x;
                    if dy < img_h as usize && dx < img_w as usize {
                        dst[dy * img_w as usize + dx] = src[y * raw_w as usize + x];
                    }
                }
            }
            Tensor::from_f32(&dst, &[1, 1, img_h, img_w], device)?
        } else {
            sample.image.unsqueeze(0)?.to_device(device)? // [1, 1, H, W]
        };
        let case = Tensor::from_f32(&[sample.case_label], &[1, 1], device)?;

        let img_cpu = img.to_device(Device::CPU)?;
        let img_var = Variable::new(img, false);
        let case_var = Variable::new(case, false);
        let origin = Variable::new(Tensor::zeros(&[1, 2], TensorOptions { device, ..Default::default() })?, false);

        let result = model.forward(&img_var, &case_var, &origin)?;

        // Predictions.
        let pred_letter = result.letter_logits.data().argmax(1, false)?
            .to_i64_vec()?[0];
        let pred_case = result.case_logits.data().argmax(1, false)?
            .to_i64_vec()?[0];

        // Reconstruction MSE (compare against model-sized input, not raw).
        let recon_data = result.recon.data().to_device(Device::CPU)?;
        let input_data = img_cpu;
        let diff = recon_data.sub(&input_data)?;
        let mse = diff.mul(&diff)?.mean()?.item()?;

        // Fixation locations.
        let scan_locs = extract_locs(&result.scan_locations)?;
        let read_locs = extract_locs(&result.read_locations)?;

        // Reconstruction pixels for atlas.
        let recon_pixels = result.recon.data().to_device(Device::CPU)?
            .squeeze(0)?.squeeze(0)?.to_f32_vec()?;

        // Latent vector for clustering analysis.
        let latent = result.latent.data().to_device(Device::CPU)?.squeeze(0)?.to_f32_vec()?;

        // Read original PNG bytes for atlas embedding.
        let png_bytes = sample.png_bytes.clone();

        let case_idx = if sample.case_str == "lower" { 1i64 } else { 0 };
        results.push(SampleResult {
            letter: sample.letter.clone(),
            letter_idx: sample.letter_idx,
            pred_letter_idx: pred_letter,
            case_str: sample.case_str.clone(),
            case_correct: pred_case == case_idx,
            font: sample.font.clone(),
            recon_mse: mse,
            scan_locs,
            read_locs,
            image_png: png_bytes,
            recon_data: recon_pixels,
            latent,
        });

        if (i + 1) % 100 == 0 || i + 1 == samples.len() {
            eprint!("\rEval: {}/{}", i + 1, samples.len());
        }
    }
    eprintln!();

    // Generate reports.
    fs::create_dir_all(&save_path)
        .map_err(|e| TensorError::new(&format!("create eval dir: {e}")))?;

    let report = generate_report(&results);
    fs::write(format!("{}/results.md", save_path), &report)
        .map_err(|e| TensorError::new(&format!("write results: {e}")))?;
    eprintln!("{}", report);

    let eval_json = generate_eval_json(&results);
    fs::write(format!("{}/eval.json", save_path), &eval_json)
        .map_err(|e| TensorError::new(&format!("write eval.json: {e}")))?;

    let atlas = generate_atlas(&results, img_h, img_w);
    fs::write(format!("{}/letter_atlas.html", save_path), &atlas)
        .map_err(|e| TensorError::new(&format!("write atlas: {e}")))?;

    // Latent space clustering analysis.
    let clustering = analyze_latent_clustering(&results);
    fs::write(format!("{}/latent_clustering.md", save_path), &clustering)
        .map_err(|e| TensorError::new(&format!("write clustering: {e}")))?;
    eprintln!("{}", clustering);

    let latent_viz = generate_latent_viz(&results);
    fs::write(format!("{}/latent_map.html", save_path), &latent_viz)
        .map_err(|e| TensorError::new(&format!("write latent map: {e}")))?;

    eprintln!("Saved: {}/results.md", save_path);
    eprintln!("Saved: {}/eval.json", save_path);
    eprintln!("Saved: {}/letter_atlas.html", save_path);
    eprintln!("Saved: {}/latent_clustering.md", save_path);
    eprintln!("Saved: {}/latent_map.html", save_path);

    Ok(())
}

/// Generate the results.md accuracy report.
fn generate_report(results: &[SampleResult]) -> String {
    let total = results.len();
    let letter_correct = results.iter()
        .filter(|r| r.pred_letter_idx == r.letter_idx)
        .count();
    let case_correct = results.iter().filter(|r| r.case_correct).count();
    let avg_mse: f64 = results.iter().map(|r| r.recon_mse).sum::<f64>() / total as f64;

    let mut s = String::with_capacity(4096);
    let _ = writeln!(s, "# Evaluation Results\n");
    let _ = writeln!(s, "| Metric | Value |");
    let _ = writeln!(s, "|--------|-------|");
    let _ = writeln!(s, "| Letter | {}/{} ({:.1}%) |", letter_correct, total,
        letter_correct as f64 / total as f64 * 100.0);
    let _ = writeln!(s, "| Case | {}/{} ({:.1}%) |", case_correct, total,
        case_correct as f64 / total as f64 * 100.0);
    let _ = writeln!(s, "| Avg MSE | {:.4} |", avg_mse);

    // Per-font breakdown.
    let mut fonts: Vec<String> = results.iter().map(|r| r.font.clone()).collect();
    fonts.sort();
    fonts.dedup();

    let _ = writeln!(s, "\n## Per-font\n");
    let _ = writeln!(s, "| Font | Letter | Case | N |");
    let _ = writeln!(s, "|------|--------|------|---|");
    for font in &fonts {
        let font_results: Vec<&SampleResult> = results.iter()
            .filter(|r| &r.font == font)
            .collect();
        let n = font_results.len();
        let fl = font_results.iter().filter(|r| r.pred_letter_idx == r.letter_idx).count();
        let fc = font_results.iter().filter(|r| r.case_correct).count();
        let _ = writeln!(s, "| {} | {:.1}% | {:.1}% | {} |",
            font, fl as f64 / n as f64 * 100.0, fc as f64 / n as f64 * 100.0, n);
    }

    // Errors (if any).
    let errors: Vec<&SampleResult> = results.iter()
        .filter(|r| r.pred_letter_idx != r.letter_idx || !r.case_correct)
        .collect();
    if errors.is_empty() {
        let _ = writeln!(s, "\nNo errors.");
    } else {
        let _ = writeln!(s, "\n## Errors ({})\n", errors.len());
        let _ = writeln!(s, "| Letter | Case | Predicted | Font |");
        let _ = writeln!(s, "|--------|------|-----------|------|");
        for r in &errors {
            let pred = idx_to_letter(r.pred_letter_idx, &r.case_str);
            let _ = writeln!(s, "| {} | {} | {} | {} |",
                r.letter, r.case_str, pred, r.font);
        }
    }

    s
}

/// Generate machine-readable eval.json.
fn generate_eval_json(results: &[SampleResult]) -> String {
    let total = results.len();
    let letter_correct = results.iter()
        .filter(|r| r.pred_letter_idx == r.letter_idx)
        .count();
    let case_correct = results.iter().filter(|r| r.case_correct).count();
    let avg_mse: f64 = results.iter().map(|r| r.recon_mse).sum::<f64>() / total as f64;

    let json = serde_json::json!({
        "total": total,
        "letter_correct": letter_correct,
        "letter_accuracy": letter_correct as f64 / total as f64,
        "case_correct": case_correct,
        "case_accuracy": case_correct as f64 / total as f64,
        "avg_recon_mse": avg_mse,
        "per_sample": results.iter().map(|r| {
            serde_json::json!({
                "letter": r.letter,
                "predicted": idx_to_letter(r.pred_letter_idx, &r.case_str),
                "letter_correct": r.pred_letter_idx == r.letter_idx,
                "case_correct": r.case_correct,
                "font": r.font,
                "recon_mse": r.recon_mse,
                "scan_locations": r.scan_locs,
                "read_locations": r.read_locs,
            })
        }).collect::<Vec<_>>(),
    });

    serde_json::to_string_pretty(&json).unwrap_or_default()
}

/// Encode bytes to base64 (no external dependency).
fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        out.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

/// Generate self-contained HTML attention atlas with navigation.
fn generate_atlas(results: &[SampleResult], img_h: i64, img_w: i64) -> String {
    // Display thumbnails at a reasonable size.
    let thumb_w: u32 = 128;
    let thumb_h = (img_h as f64 * thumb_w as f64 / img_w as f64) as u32;
    let fw = thumb_w as f64;
    let fh = thumb_h as f64;

    let total = results.len();
    let letter_correct = results.iter()
        .filter(|r| r.pred_letter_idx == r.letter_idx).count();
    let case_correct = results.iter().filter(|r| r.case_correct).count();

    // Collect unique fonts (sorted).
    let mut fonts: Vec<String> = results.iter().map(|r| r.font.clone()).collect();
    fonts.sort();
    fonts.dedup();

    // Group results by letter string.
    let mut by_letter: Vec<(&str, Vec<&SampleResult>)> = Vec::new();
    let mut current_letter = "";
    for r in results {
        if r.letter.as_str() != current_letter {
            current_letter = &r.letter;
            by_letter.push((&r.letter, Vec::new()));
        }
        by_letter.last_mut().unwrap().1.push(r);
    }

    let mut html = String::with_capacity(2 * 1024 * 1024);

    // Header + CSS + JS.
    html.push_str("<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\">\n");
    html.push_str("<title>Letter Attention Atlas</title>\n");
    html.push_str("<style>\n");
    html.push_str("*{box-sizing:border-box}\n");
    html.push_str("body{font-family:system-ui,-apple-system,sans-serif;background:#1a1a2e;color:#e0e0e0;margin:0;padding:20px}\n");
    html.push_str("h1{color:#fff;margin:0 0 4px}\n");
    html.push_str(".summary{color:#aaa;margin-bottom:12px}\n");
    html.push_str(".toolbar{position:sticky;top:0;z-index:10;background:#1a1a2e;padding:8px 0 12px;border-bottom:1px solid #333}\n");
    html.push_str(".nav{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px}\n");
    html.push_str(".nav a{display:inline-block;padding:4px 8px;background:#252540;color:#ccc;text-decoration:none;border-radius:4px;font-size:13px;min-width:28px;text-align:center}\n");
    html.push_str(".nav a:hover,.nav a.active{background:#3498db;color:#fff}\n");
    html.push_str(".nav a.has-error{border:1px solid #e74c3c}\n");
    html.push_str(".filters{display:flex;flex-wrap:wrap;gap:6px;align-items:center;font-size:12px}\n");
    html.push_str(".filters label{cursor:pointer;color:#aaa}\n");
    html.push_str(".filters label:hover{color:#fff}\n");
    html.push_str(".filters input{margin-right:2px}\n");
    let _ = writeln!(html, ".btn{{padding:3px 8px;background:#252540;color:#aaa;border:1px solid #444;border-radius:4px;cursor:pointer;font-size:11px}}");
    html.push_str(".btn:hover{background:#333;color:#fff}\n");
    html.push_str(".btn.active{background:#e74c3c;color:#fff;border-color:#e74c3c}\n");
    html.push_str(".letter-section{margin-top:16px}\n");
    html.push_str(".letter-section h2{color:#ccc;margin:0 0 8px;padding-bottom:4px;border-bottom:1px solid #333;font-size:16px}\n");
    html.push_str(".letter-row{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px}\n");
    html.push_str(".sample{display:inline-block;text-align:center;background:#252540;border-radius:6px;padding:4px}\n");
    html.push_str(".sample.error{border:2px solid #e74c3c}\n");
    html.push_str(".sample.hidden{display:none}\n");
    html.push_str(".label{font-size:10px;color:#888;margin-top:1px}\n");
    html.push_str(".pred{font-size:10px;font-weight:bold}\n");
    html.push_str(".pred.correct{color:#2ecc71} .pred.wrong{color:#e74c3c}\n");
    html.push_str(".pair{display:flex;gap:2px}\n");
    html.push_str(".legend{margin:8px 0;font-size:12px;color:#888}\n");
    html.push_str(".legend span{margin-right:14px}\n");
    html.push_str("</style></head><body>\n");

    // Summary.
    let _ = writeln!(html, "<h1>Letter Attention Atlas</h1>");
    let _ = writeln!(html, "<div class=\"summary\">Letter: {}/{} ({:.1}%) &middot; Case: {}/{} ({:.1}%) &middot; {}×{} &middot; {} fonts</div>",
        letter_correct, total, letter_correct as f64 / total as f64 * 100.0,
        case_correct, total, case_correct as f64 / total as f64 * 100.0,
        img_h, img_w, fonts.len());

    // Toolbar: letter nav + font filter + errors toggle.
    html.push_str("<div class=\"toolbar\">\n");

    // Letter nav.
    html.push_str("<div class=\"nav\" id=\"letter-nav\">\n");
    for (letter, samples) in &by_letter {
        let has_error = samples.iter().any(|r| r.pred_letter_idx != r.letter_idx || !r.case_correct);
        let err_class = if has_error { " has-error" } else { "" };
        let _ = writeln!(html, "<a href=\"#letter-{}\" class=\"{}\">{}</a>", letter, err_class, letter);
    }
    html.push_str("</div>\n");

    // Filters row.
    html.push_str("<div class=\"filters\">\n");
    html.push_str("<button class=\"btn\" id=\"errors-btn\" onclick=\"toggleErrors()\">Errors only</button>\n");
    html.push_str("<span style=\"color:#555\">|</span>\n");
    for font in &fonts {
        let font_id = font.replace(|c: char| !c.is_alphanumeric(), "_");
        let _ = writeln!(html, "<label><input type=\"checkbox\" checked onchange=\"filterFont()\" data-font=\"{}\"> {}</label>",
            font_id, font);
    }
    html.push_str("</div>\n");

    // Legend.
    html.push_str("<div class=\"legend\">\n");
    html.push_str("<span><svg width=\"10\" height=\"10\"><circle cx=\"5\" cy=\"5\" r=\"4\" fill=\"#3498db\"/></svg> Scan</span>\n");
    html.push_str("<span><svg width=\"10\" height=\"10\"><circle cx=\"5\" cy=\"5\" r=\"4\" fill=\"#e74c3c\"/></svg> Read</span>\n");
    html.push_str("<span>Left: input + fixations &middot; Right: reconstruction</span>\n");
    html.push_str("</div>\n");
    html.push_str("</div>\n"); // .toolbar

    // Per-letter sections.
    for (letter, samples) in &by_letter {
        let _ = writeln!(html, "<div class=\"letter-section\" id=\"letter-{}\">", letter);
        let _ = writeln!(html, "<h2>{}</h2>\n<div class=\"letter-row\">", letter);

        for r in samples {
            let is_correct = r.pred_letter_idx == r.letter_idx && r.case_correct;
            let pred = idx_to_letter(r.pred_letter_idx, &r.case_str);
            let error_class = if is_correct { "" } else { " error" };
            let font_id = r.font.replace(|c: char| !c.is_alphanumeric(), "_");

            let _ = writeln!(html, "<div class=\"sample{}\" data-font=\"{}\" data-correct=\"{}\">",
                error_class, font_id, is_correct);
            html.push_str("<div class=\"pair\">\n");

            // Input image with fixation overlay (SVG).
            let _ = writeln!(html, "<svg width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">",
                thumb_w, thumb_h, thumb_w, thumb_h);
            if !r.image_png.is_empty() {
                let img_b64 = base64_encode(&r.image_png);
                let _ = writeln!(html, "<image href=\"data:image/png;base64,{}\" width=\"{}\" height=\"{}\"/>",
                    img_b64, thumb_w, thumb_h);
            }
            let all_locs: Vec<([f64; 2], bool)> = r.scan_locs.iter().map(|l| (*l, true))
                .chain(r.read_locs.iter().map(|l| (*l, false))).collect();

            for pair in all_locs.windows(2) {
                let (p0, _) = pair[0];
                let (p1, _) = pair[1];
                let x0 = (p0[0] + 1.0) / 2.0 * fw;
                let y0 = (p0[1] + 1.0) / 2.0 * fh;
                let x1 = (p1[0] + 1.0) / 2.0 * fw;
                let y1 = (p1[1] + 1.0) / 2.0 * fh;
                let _ = writeln!(html, "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#fff\" stroke-width=\"0.8\" opacity=\"0.35\"/>",
                    x0, y0, x1, y1);
            }

            for (i, loc) in r.scan_locs.iter().enumerate() {
                let px = (loc[0] + 1.0) / 2.0 * fw;
                let py = (loc[1] + 1.0) / 2.0 * fh;
                let _ = writeln!(html, "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"4\" fill=\"#3498db\" opacity=\"0.8\"/>", px, py);
                let _ = writeln!(html, "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"#fff\" font-size=\"6\" text-anchor=\"middle\" dominant-baseline=\"central\">S{}</text>",
                    px, py, i + 1);
            }
            for (i, loc) in r.read_locs.iter().enumerate() {
                let px = (loc[0] + 1.0) / 2.0 * fw;
                let py = (loc[1] + 1.0) / 2.0 * fh;
                let _ = writeln!(html, "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"3\" fill=\"#e74c3c\" opacity=\"0.8\"/>", px, py);
                let _ = writeln!(html, "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"#fff\" font-size=\"6\" text-anchor=\"middle\" dominant-baseline=\"central\">{}</text>",
                    px, py, i + 1);
            }
            html.push_str("</svg>\n");

            // Reconstruction.
            let recon_pixels = (img_h * img_w) as usize;
            if r.recon_data.len() == recon_pixels {
                let recon_png = pixels_to_png(&r.recon_data, img_w as usize, img_h as usize);
                let recon_b64 = base64_encode(&recon_png);
                let _ = writeln!(html, "<img src=\"data:image/png;base64,{}\" width=\"{}\" height=\"{}\">",
                    recon_b64, thumb_w, thumb_h);
            }

            html.push_str("</div>\n"); // .pair
            let _ = writeln!(html, "<div class=\"label\">{}</div>", r.font);
            let correct_class = if is_correct { "correct" } else { "wrong" };
            let _ = writeln!(html, "<div class=\"pred {}\">{}</div>", correct_class, pred);
            html.push_str("</div>\n"); // .sample
        }
        html.push_str("</div></div>\n"); // .letter-row .letter-section
    }

    // JavaScript for filtering.
    html.push_str("<script>\n");
    html.push_str("let errorsOnly = false;\n");
    html.push_str("function toggleErrors() {\n");
    html.push_str("  errorsOnly = !errorsOnly;\n");
    html.push_str("  document.getElementById('errors-btn').classList.toggle('active', errorsOnly);\n");
    html.push_str("  applyFilters();\n");
    html.push_str("}\n");
    html.push_str("function filterFont() { applyFilters(); }\n");
    html.push_str("function applyFilters() {\n");
    html.push_str("  const checked = new Set();\n");
    html.push_str("  document.querySelectorAll('.filters input[type=checkbox]').forEach(cb => {\n");
    html.push_str("    if (cb.checked) checked.add(cb.dataset.font);\n");
    html.push_str("  });\n");
    html.push_str("  document.querySelectorAll('.sample').forEach(el => {\n");
    html.push_str("    const fontMatch = checked.has(el.dataset.font);\n");
    html.push_str("    const errMatch = !errorsOnly || el.dataset.correct === 'false';\n");
    html.push_str("    el.classList.toggle('hidden', !(fontMatch && errMatch));\n");
    html.push_str("  });\n");
    html.push_str("}\n");
    html.push_str("</script>\n");
    html.push_str("</body></html>");
    html
}

// --- Latent space analysis ---

fn cosine_dist(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-12 { 1.0 } else { 1.0 - dot / denom }
}

fn l2_dist(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b.iter())
        .map(|(&x, &y)| ((x - y) as f64).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Silhouette score to color: red(low) -> yellow(mid) -> green(high).
fn sil_color(s: f64) -> &'static str {
    if s >= 0.5 { "#4caf50" }       // green
    else if s >= 0.4 { "#8bc34a" }  // light green
    else if s >= 0.3 { "#ffc107" }  // amber
    else if s >= 0.2 { "#ff9800" }  // orange
    else { "#f44336" }              // red
}

/// Analyze latent space clustering: intra-letter vs inter-letter distances.
fn analyze_latent_clustering(results: &[SampleResult]) -> String {
    // Group latents by (letter_idx, case).
    let mut by_letter: HashMap<(i64, &str), Vec<&[f32]>> = HashMap::new();
    for r in results {
        by_letter.entry((r.letter_idx, r.case_str.as_str()))
            .or_default()
            .push(&r.latent);
    }

    // Also group by letter only (both cases).
    let mut by_letter_only: HashMap<i64, Vec<&[f32]>> = HashMap::new();
    for r in results {
        by_letter_only.entry(r.letter_idx)
            .or_default()
            .push(&r.latent);
    }

    // Compute centroids per letter.
    let dim = results[0].latent.len();
    let mut centroids: HashMap<i64, Vec<f32>> = HashMap::new();
    for (&idx, vecs) in &by_letter_only {
        let mut centroid = vec![0.0f32; dim];
        for v in vecs {
            for (c, &x) in centroid.iter_mut().zip(v.iter()) {
                *c += x;
            }
        }
        let n = vecs.len() as f32;
        for c in &mut centroid { *c /= n; }
        centroids.insert(idx, centroid);
    }

    // Intra-letter: average pairwise L2 within each letter class.
    let mut intra_dists: Vec<(String, f64)> = Vec::new();
    let mut total_intra = 0.0;
    let mut intra_count = 0;
    for (&idx, vecs) in &by_letter_only {
        let mut sum = 0.0;
        let mut n = 0;
        for i in 0..vecs.len() {
            for j in (i+1)..vecs.len() {
                sum += l2_dist(vecs[i], vecs[j]);
                n += 1;
            }
        }
        if n > 0 {
            let avg = sum / n as f64;
            let ch = LETTERS[idx as usize];
            intra_dists.push((format!("{}", ch), avg));
            total_intra += sum;
            intra_count += n;
        }
    }

    // Inter-letter: average L2 between centroids of different letters.
    let letter_idxs: Vec<i64> = centroids.keys().copied().collect();
    let mut total_inter = 0.0;
    let mut inter_count = 0;
    let mut closest_pairs: Vec<(String, String, f64)> = Vec::new();
    for i in 0..letter_idxs.len() {
        for j in (i+1)..letter_idxs.len() {
            let d = l2_dist(&centroids[&letter_idxs[i]], &centroids[&letter_idxs[j]]);
            total_inter += d;
            inter_count += 1;
            closest_pairs.push((
                format!("{}", LETTERS[letter_idxs[i] as usize]),
                format!("{}", LETTERS[letter_idxs[j] as usize]),
                d,
            ));
        }
    }
    closest_pairs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    // Case separation: distance between upper/lower centroids for same letter.
    let mut case_dists: Vec<(String, f64)> = Vec::new();
    for idx in 0..26i64 {
        if let (Some(upper), Some(lower)) = (
            by_letter.get(&(idx, "upper")),
            by_letter.get(&(idx, "lower")),
        ) {
            let mut cu = vec![0.0f32; dim];
            for v in upper { for (c, &x) in cu.iter_mut().zip(v.iter()) { *c += x; } }
            let nu = upper.len() as f32;
            for c in &mut cu { *c /= nu; }

            let mut cl = vec![0.0f32; dim];
            for v in lower { for (c, &x) in cl.iter_mut().zip(v.iter()) { *c += x; } }
            let nl = lower.len() as f32;
            for c in &mut cl { *c /= nl; }

            case_dists.push((format!("{}", LETTERS[idx as usize]), l2_dist(&cu, &cl)));
        }
    }

    // Format report.
    let avg_intra = if intra_count > 0 { total_intra / intra_count as f64 } else { 0.0 };
    let avg_inter = if inter_count > 0 { total_inter / inter_count as f64 } else { 0.0 };
    let ratio = if avg_inter > 0.0 { avg_intra / avg_inter } else { f64::INFINITY };

    let mut s = String::with_capacity(4096);
    let _ = writeln!(s, "# Latent Space Clustering\n");
    let _ = writeln!(s, "| Metric | Value |");
    let _ = writeln!(s, "|--------|-------|");
    let _ = writeln!(s, "| Avg intra-letter L2 | {:.4} |", avg_intra);
    let _ = writeln!(s, "| Avg inter-letter L2 (centroids) | {:.4} |", avg_inter);
    let _ = writeln!(s, "| Ratio (intra/inter) | {:.4} |", ratio);
    let _ = writeln!(s, "| Latent dim | {} |", dim);
    let _ = writeln!(s, "| Samples | {} |", results.len());

    let _ = writeln!(s, "\n## Closest letter pairs (centroid L2)\n");
    let _ = writeln!(s, "| Pair | Distance |");
    let _ = writeln!(s, "|------|----------|");
    for (a, b, d) in closest_pairs.iter().take(15) {
        let _ = writeln!(s, "| {}-{} | {:.4} |", a, b, d);
    }

    let _ = writeln!(s, "\n## Case separation (upper↔lower centroid L2)\n");
    let _ = writeln!(s, "| Letter | Distance |");
    let _ = writeln!(s, "|--------|----------|");
    case_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for (ch, d) in &case_dists {
        let _ = writeln!(s, "| {} | {:.4} |", ch, d);
    }

    intra_dists.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let _ = writeln!(s, "\n## Intra-letter spread (avg pairwise L2)\n");
    let _ = writeln!(s, "| Letter | Spread |");
    let _ = writeln!(s, "|--------|--------|");
    for (ch, d) in &intra_dists {
        let _ = writeln!(s, "| {} | {:.4} |", ch, d);
    }

    // Silhouette score per letter+case.
    // Groups are (letter_idx, case). For each sample:
    //   a = mean L2 to other samples in same group
    //   b = mean L2 to samples in nearest other group
    //   silhouette = (b - a) / max(a, b)
    let groups: Vec<(i64, &str)> = by_letter.keys().copied().collect();
    // Precompute group centroids for fast nearest-group lookup.
    let mut group_centroids: HashMap<(i64, &str), Vec<f32>> = HashMap::new();
    for (&key, vecs) in &by_letter {
        let mut c = vec![0.0f32; dim];
        for v in vecs { for (ci, &x) in c.iter_mut().zip(v.iter()) { *ci += x; } }
        let n = vecs.len() as f32;
        for ci in &mut c { *ci /= n; }
        group_centroids.insert(key, c);
    }

    let mut sil_by_group: Vec<(String, String, f64, usize)> = Vec::new();
    for &gk in &groups {
        let own = &by_letter[&gk];
        if own.len() < 2 { continue; }

        let mut group_sil_sum = 0.0;
        for (i, sample) in own.iter().enumerate() {
            // a: mean distance to other members of same group.
            let a: f64 = own.iter().enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, other)| l2_dist(sample, other))
                .sum::<f64>() / (own.len() - 1) as f64;

            // b: mean distance to nearest other group (by centroid distance).
            let mut best_b = f64::INFINITY;
            for &ok in &groups {
                if ok == gk { continue; }
                let other_vecs = &by_letter[&ok];
                let mean_d: f64 = other_vecs.iter()
                    .map(|o| l2_dist(sample, o))
                    .sum::<f64>() / other_vecs.len() as f64;
                if mean_d < best_b { best_b = mean_d; }
            }

            let sil = if a.max(best_b) > 0.0 { (best_b - a) / a.max(best_b) } else { 0.0 };
            group_sil_sum += sil;
        }
        let avg_sil = group_sil_sum / own.len() as f64;
        let ch = LETTERS[gk.0 as usize];
        sil_by_group.push((format!("{}", ch), gk.1.to_string(), avg_sil, own.len()));
    }
    sil_by_group.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    let total_sil: f64 = sil_by_group.iter().map(|g| g.2 * g.3 as f64).sum();
    let total_n: usize = sil_by_group.iter().map(|g| g.3).sum();
    let avg_sil = if total_n > 0 { total_sil / total_n as f64 } else { 0.0 };

    let _ = writeln!(s, "\n## Silhouette score per letter+case\n");
    let _ = writeln!(s, "Silhouette: -1 (wrong cluster) to +1 (perfectly separated). Overall: **{:.4}**\n", avg_sil);
    let _ = writeln!(s, "| Letter | Case | Silhouette | N |");
    let _ = writeln!(s, "|--------|------|------------|---|");
    for (ch, cas, sil, n) in &sil_by_group {
        let _ = writeln!(s, "| {} | {} | {:.4} | {} |", ch, cas, sil, n);
    }

    // --- Latent capacity analysis ---
    // Center the data.
    let n = results.len();
    let mut mean = vec![0.0f64; dim];
    for r in results {
        for (m, &x) in mean.iter_mut().zip(r.latent.iter()) {
            *m += x as f64;
        }
    }
    for m in &mut mean { *m /= n as f64; }

    let mut centered: Vec<Vec<f64>> = results.iter()
        .map(|r| r.latent.iter().zip(mean.iter()).map(|(&x, &m)| x as f64 - m).collect())
        .collect();

    // Extract top eigenvalues via successive power iteration + deflation.
    let num_eigenvalues = dim.min(50); // top 50 is enough to see the spectrum
    let mut eigenvalues: Vec<f64> = Vec::with_capacity(num_eigenvalues);
    for _ in 0..num_eigenvalues {
        let ev = power_iteration(&centered, dim, 100);
        // Eigenvalue = ||X v||^2 / n (variance along this direction).
        let mut lambda = 0.0;
        for row in centered.iter() {
            let proj: f64 = row.iter().zip(ev.iter()).map(|(&a, &b)| a * b).sum();
            lambda += proj * proj;
        }
        lambda /= n as f64;
        if lambda < 1e-10 { break; } // spectrum died
        eigenvalues.push(lambda);
        // Deflate.
        for row in &mut centered {
            let proj: f64 = row.iter().zip(ev.iter()).map(|(&x, &p)| x * p).sum();
            for (x, &p) in row.iter_mut().zip(ev.iter()) {
                *x -= proj * p;
            }
        }
    }

    let total_var: f64 = eigenvalues.iter().sum();
    // Also compute residual variance (what's left after top-k).
    let mut residual_var = 0.0;
    for row in &centered {
        for &x in row { residual_var += x * x; }
    }
    residual_var /= n as f64;
    let full_var = total_var + residual_var;

    let _ = writeln!(s, "\n## Latent Capacity Analysis\n");
    let _ = writeln!(s, "Eigenvalue spectrum of latent covariance (top {} of {} dims).\n", eigenvalues.len(), dim);
    let _ = writeln!(s, "| Metric | Value |");
    let _ = writeln!(s, "|--------|-------|");
    let _ = writeln!(s, "| Total variance | {:.2} |", full_var);
    let _ = writeln!(s, "| Top {} variance | {:.2} ({:.1}%) |", eigenvalues.len(), total_var, total_var / full_var * 100.0);
    let _ = writeln!(s, "| Residual ({} dims) | {:.2} ({:.1}%) |", dim - eigenvalues.len(), residual_var, residual_var / full_var * 100.0);

    // Cumulative explained variance thresholds.
    let mut cumsum = 0.0;
    let mut dims_90 = 0;
    let mut dims_95 = 0;
    let mut dims_99 = 0;
    for (i, &ev) in eigenvalues.iter().enumerate() {
        cumsum += ev;
        if dims_90 == 0 && cumsum / full_var >= 0.90 { dims_90 = i + 1; }
        if dims_95 == 0 && cumsum / full_var >= 0.95 { dims_95 = i + 1; }
        if dims_99 == 0 && cumsum / full_var >= 0.99 { dims_99 = i + 1; }
    }
    let _ = writeln!(s, "| Dims for 90% variance | {} |", if dims_90 > 0 { dims_90.to_string() } else { format!(">{}", eigenvalues.len()) });
    let _ = writeln!(s, "| Dims for 95% variance | {} |", if dims_95 > 0 { dims_95.to_string() } else { format!(">{}", eigenvalues.len()) });
    let _ = writeln!(s, "| Dims for 99% variance | {} |", if dims_99 > 0 { dims_99.to_string() } else { format!(">{}", eigenvalues.len()) });

    // Active dims: eigenvalues > 1% of top eigenvalue.
    let top_ev = eigenvalues.first().copied().unwrap_or(1.0);
    let active_dims = eigenvalues.iter().filter(|&&v| v > top_ev * 0.01).count();
    let _ = writeln!(s, "| Active dims (>1% of top) | {} / {} |", active_dims, dim);

    // Spectrum: show top eigenvalues with bar chart.
    let _ = writeln!(s, "\n### Eigenvalue spectrum\n");
    let _ = writeln!(s, "| # | Eigenvalue | % Var | Cumul % | |");
    let _ = writeln!(s, "|---|-----------|-------|---------|--|");
    cumsum = 0.0;
    let bar_max = eigenvalues.first().copied().unwrap_or(1.0);
    for (i, &ev) in eigenvalues.iter().enumerate() {
        cumsum += ev;
        let pct = ev / full_var * 100.0;
        let cum_pct = cumsum / full_var * 100.0;
        let bar_len = (ev / bar_max * 30.0) as usize;
        let bar: String = std::iter::repeat('█').take(bar_len).collect();
        let _ = writeln!(s, "| {} | {:.2} | {:.1}% | {:.1}% | {} |", i + 1, ev, pct, cum_pct, bar);
    }

    s
}

/// Generate self-contained HTML with 2D PCA projection of latent space.
/// PCA is simpler than t-SNE, runs in Rust, and good enough for cluster structure.
fn generate_latent_viz(results: &[SampleResult]) -> String {
    let n = results.len();
    let dim = results[0].latent.len();

    // Compute mean.
    let mut mean = vec![0.0f64; dim];
    for r in results {
        for (m, &x) in mean.iter_mut().zip(r.latent.iter()) {
            *m += x as f64;
        }
    }
    for m in &mut mean { *m /= n as f64; }

    // Center the data.
    let mut centered: Vec<Vec<f64>> = results.iter()
        .map(|r| r.latent.iter().zip(mean.iter()).map(|(&x, &m)| x as f64 - m).collect())
        .collect();

    // Power iteration for top 2 principal components.
    let pc1 = power_iteration(&centered, dim, 100);
    // Deflate.
    for row in &mut centered {
        let proj: f64 = row.iter().zip(pc1.iter()).map(|(&x, &p)| x * p).sum();
        for (x, &p) in row.iter_mut().zip(pc1.iter()) {
            *x -= proj * p;
        }
    }
    let pc2 = power_iteration(&centered, dim, 100);

    // Re-center (undo deflation for projection).
    let centered_orig: Vec<Vec<f64>> = results.iter()
        .map(|r| r.latent.iter().zip(mean.iter()).map(|(&x, &m)| x as f64 - m).collect())
        .collect();

    // Project onto PC1, PC2.
    let mut points: Vec<(f64, f64, i64, String, String, String)> = Vec::with_capacity(n);
    for (i, r) in results.iter().enumerate() {
        let x: f64 = centered_orig[i].iter().zip(pc1.iter()).map(|(&a, &b)| a * b).sum();
        let y: f64 = centered_orig[i].iter().zip(pc2.iter()).map(|(&a, &b)| a * b).sum();
        points.push((x, y, r.letter_idx, r.case_str.clone(), r.letter.clone(), r.font.clone()));
    }

    // Compute silhouette scores per (letter_idx, case) group.
    let mut by_group: HashMap<(i64, &str), Vec<&[f32]>> = HashMap::new();
    for r in results {
        by_group.entry((r.letter_idx, r.case_str.as_str()))
            .or_default()
            .push(&r.latent);
    }
    let group_keys: Vec<(i64, &str)> = by_group.keys().copied().collect();
    let mut sil_scores: HashMap<String, f64> = HashMap::new();
    for &gk in &group_keys {
        let own = &by_group[&gk];
        if own.len() < 2 { continue; }
        let mut group_sum = 0.0;
        for (i, sample) in own.iter().enumerate() {
            let a: f64 = own.iter().enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, o)| l2_dist(sample, o))
                .sum::<f64>() / (own.len() - 1) as f64;
            let mut best_b = f64::INFINITY;
            for &ok in &group_keys {
                if ok == gk { continue; }
                let ov = &by_group[&ok];
                let mean_d: f64 = ov.iter().map(|o| l2_dist(sample, o)).sum::<f64>() / ov.len() as f64;
                if mean_d < best_b { best_b = mean_d; }
            }
            let sil = if a.max(best_b) > 0.0 { (best_b - a) / a.max(best_b) } else { 0.0 };
            group_sum += sil;
        }
        let key = format!("{}_{}", gk.0, gk.1);
        sil_scores.insert(key, group_sum / own.len() as f64);
    }
    // Sort for top/bottom 5.
    let mut sil_sorted: Vec<(i64, String, f64)> = group_keys.iter()
        .filter_map(|&(idx, cas)| {
            sil_scores.get(&format!("{}_{}", idx, cas)).map(|&s| (idx, cas.to_string(), s))
        })
        .collect();
    sil_sorted.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    let total_sil: f64 = sil_sorted.iter().map(|g| g.2).sum::<f64>();
    let avg_sil = if !sil_sorted.is_empty() { total_sil / sil_sorted.len() as f64 } else { 0.0 };

    // Aggregate silhouette for all-upper and all-lower.
    let upper_entries: Vec<&(i64, String, f64)> = sil_sorted.iter().filter(|g| g.1 == "upper").collect();
    let lower_entries: Vec<&(i64, String, f64)> = sil_sorted.iter().filter(|g| g.1 == "lower").collect();
    let upper_sil = if !upper_entries.is_empty() {
        upper_entries.iter().map(|g| g.2).sum::<f64>() / upper_entries.len() as f64
    } else { 0.0 };
    let lower_sil = if !lower_entries.is_empty() {
        lower_entries.iter().map(|g| g.2).sum::<f64>() / lower_entries.len() as f64
    } else { 0.0 };

    // --- Latent capacity: eigenvalue spectrum via power iteration ---
    let mut cap_centered: Vec<Vec<f64>> = results.iter()
        .map(|r| r.latent.iter().zip(mean.iter()).map(|(&x, &m)| x as f64 - m).collect())
        .collect();
    let num_ev = dim.min(50);
    let mut eigenvalues: Vec<f64> = Vec::with_capacity(num_ev);
    for _ in 0..num_ev {
        let ev = power_iteration(&cap_centered, dim, 100);
        let mut lambda = 0.0;
        for row in cap_centered.iter() {
            let proj: f64 = row.iter().zip(ev.iter()).map(|(&a, &b)| a * b).sum();
            lambda += proj * proj;
        }
        lambda /= n as f64;
        if lambda < 1e-10 { break; }
        eigenvalues.push(lambda);
        for row in &mut cap_centered {
            let proj: f64 = row.iter().zip(ev.iter()).map(|(&x, &p)| x * p).sum();
            for (x, &p) in row.iter_mut().zip(ev.iter()) {
                *x -= proj * p;
            }
        }
    }
    let cap_top_var: f64 = eigenvalues.iter().sum();
    let mut cap_residual = 0.0;
    for row in &cap_centered {
        for &x in row { cap_residual += x * x; }
    }
    cap_residual /= n as f64;
    let cap_full_var = cap_top_var + cap_residual;

    let mut cumsum = 0.0;
    let mut dims_90 = 0usize;
    let mut dims_95 = 0usize;
    let mut dims_99 = 0usize;
    for (i, &ev) in eigenvalues.iter().enumerate() {
        cumsum += ev;
        if dims_90 == 0 && cumsum / cap_full_var >= 0.90 { dims_90 = i + 1; }
        if dims_95 == 0 && cumsum / cap_full_var >= 0.95 { dims_95 = i + 1; }
        if dims_99 == 0 && cumsum / cap_full_var >= 0.99 { dims_99 = i + 1; }
    }
    let top_ev = eigenvalues.first().copied().unwrap_or(1.0);
    let active_dims = eigenvalues.iter().filter(|&&v| v > top_ev * 0.01).count();

    // Build HTML.
    let mut html = String::with_capacity(256 * 1024);
    html.push_str("<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\">\n");
    html.push_str("<title>Latent Space Map</title>\n");
    html.push_str("<style>\n");
    html.push_str("body{font-family:system-ui;background:#1a1a2e;color:#e0e0e0;margin:20px}\n");
    html.push_str("h1{color:#fff;margin:0 0 8px}\n");
    html.push_str(".info{color:#aaa;margin-bottom:12px;font-size:13px}\n");
    html.push_str("canvas{background:#252540;border-radius:8px;cursor:crosshair;width:100%;min-width:800px;display:block}\n");
    html.push_str(".tooltip{position:absolute;background:#333;color:#fff;padding:6px 10px;border-radius:4px;font-size:12px;pointer-events:none;display:none;line-height:1.5;z-index:10}\n");
    html.push_str(".controls{margin-top:12px;display:flex;align-items:center;gap:16px;font-size:13px}\n");
    html.push_str(".sel{color:#fff;font-weight:bold;font-size:15px}\n");
    // Summary bar
    html.push_str(".summary{display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap}\n");
    html.push_str(".scard{background:#252540;border-radius:8px;padding:12px 16px;flex:1;min-width:140px}\n");
    html.push_str(".scard h3{color:#888;font-size:11px;text-transform:uppercase;letter-spacing:0.5px;margin:0 0 4px}\n");
    html.push_str(".scard .val{font-size:22px;font-weight:bold}\n");
    html.push_str(".scard.wide{flex:2.5;min-width:280px}\n");
    html.push_str(".scard table{width:100%;border-collapse:collapse;margin-top:6px}\n");
    html.push_str(".scard td{padding:2px 6px;font-size:12px}\n");
    html.push_str(".scard tr[data-idx]{cursor:pointer;border-radius:3px}\n");
    html.push_str(".scard tr[data-idx]:hover{background:#333}\n");
    html.push_str(".scard .bar{height:6px;border-radius:3px;display:inline-block;vertical-align:middle}\n");
    html.push_str(".scard .metric-help{color:#666;font-size:10px;margin-top:8px;line-height:1.4}\n");
    // Letter bar
    html.push_str(".lbar{margin-bottom:12px}\n");
    html.push_str(".lrow{display:flex;gap:2px;align-items:center;margin-bottom:3px}\n");
    html.push_str(".lrow .lbl{font-size:11px;color:#888;width:42px;flex-shrink:0}\n");
    html.push_str(".lbtn{width:28px;height:28px;border:none;border-radius:4px;cursor:pointer;font-size:13px;font-weight:bold;text-align:center;line-height:28px;padding:0;transition:transform 0.1s}\n");
    html.push_str(".lbtn:hover{transform:scale(1.3);z-index:1}\n");
    html.push_str(".lbtn.active{outline:2px solid #fff;outline-offset:1px}\n");
    html.push_str(".ebar{display:flex;align-items:center;gap:6px;margin-bottom:12px;background:#2a1a1a;border:1px solid #442;border-radius:8px;padding:8px 12px}\n");
    html.push_str(".ebar-label{font-size:12px;color:#f44336;font-weight:bold;margin-right:4px}\n");
    html.push_str(".ebtn{border:2px solid #f44336!important}\n");
    html.push_str(".ebar-details{font-size:11px;color:#ccc;margin-left:8px;display:flex;gap:12px}\n");
    html.push_str("</style></head><body>\n");
    html.push_str("<h1>Latent Space Map (PCA)</h1>\n");
    html.push_str("<div class=\"info\">Each dot = one sample. Color = letter identity. Hover for details. Click letter buttons or dots to select.</div>\n");

    // --- Summary row 1: scores ---
    html.push_str("<div class=\"summary\">\n");
    let _ = write!(html, "<div class=\"scard\"><h3>Overall Silhouette</h3><div class=\"val\" style=\"color:{}\">{:.3}</div>", sil_color(avg_sil), avg_sil);
    html.push_str("<div class=\"metric-help\">How well letter+case groups separate in 256-dim latent space. +1 = perfect, 0 = boundary, -1 = wrong cluster.</div>");
    html.push_str("</div>\n");
    let _ = write!(html, "<div class=\"scard\"><h3>Upper Case</h3><div class=\"val\" style=\"color:{}\">{:.3}</div></div>\n", sil_color(upper_sil), upper_sil);
    let _ = write!(html, "<div class=\"scard\"><h3>Lower Case</h3><div class=\"val\" style=\"color:{}\">{:.3}</div></div>\n", sil_color(lower_sil), lower_sil);

    // Capacity card
    let cap_color = if active_dims > dim * 80 / 100 { "#f44336" }
        else if active_dims > dim * 60 / 100 { "#ff9800" }
        else { "#4caf50" };
    html.push_str("<div class=\"scard\">");
    html.push_str("<h3>Latent Capacity</h3>");
    let _ = write!(html, "<div class=\"val\" style=\"color:{}\">{} / {}</div>", cap_color, active_dims, dim);
    html.push_str("<div class=\"metric-help\">");
    let _ = write!(html, "Active dims (&gt;1% of top eigenvalue)<br>");
    let _ = write!(html, "90% var: {} dims<br>", if dims_90 > 0 { dims_90.to_string() } else { format!(">{}", eigenvalues.len()) });
    let _ = write!(html, "95% var: {} dims<br>", if dims_95 > 0 { dims_95.to_string() } else { format!(">{}", eigenvalues.len()) });
    let _ = write!(html, "99% var: {} dims", if dims_99 > 0 { dims_99.to_string() } else { format!(">{}", eigenvalues.len()) });
    html.push_str("</div></div>\n");

    html.push_str("</div>\n");

    // --- Summary row 2: top/bottom 5 ---
    html.push_str("<div class=\"summary\">\n");
    html.push_str("<div class=\"scard wide\"><h3>Least Clustered</h3><table>\n");
    for &(idx, ref cas, sil) in sil_sorted.iter().take(5) {
        let ch = LETTERS[idx as usize];
        let _ = writeln!(html,
            "<tr data-idx=\"{}\" data-cas=\"{}\"><td><b>{}</b></td><td>{}</td><td style=\"color:{}\">{:.3}</td><td><span class=\"bar\" style=\"width:{}px;background:{}\"></span></td></tr>",
            idx, cas, ch, cas, sil_color(sil), sil, (sil.max(0.0) * 100.0) as u32, sil_color(sil));
    }
    html.push_str("</table></div>\n");
    html.push_str("<div class=\"scard wide\"><h3>Best Clustered</h3><table>\n");
    for &(idx, ref cas, sil) in sil_sorted.iter().rev().take(5) {
        let ch = LETTERS[idx as usize];
        let _ = writeln!(html,
            "<tr data-idx=\"{}\" data-cas=\"{}\"><td><b>{}</b></td><td>{}</td><td style=\"color:{}\">{:.3}</td><td><span class=\"bar\" style=\"width:{}px;background:{}\"></span></td></tr>",
            idx, cas, ch, cas, sil_color(sil), sil, (sil.max(0.0) * 100.0) as u32, sil_color(sil));
    }
    html.push_str("</table></div>\n");
    html.push_str("</div>\n");

    // --- Error bar: only shown when there are misclassifications ---
    // Collect unique (idx, case) pairs involved in errors (both true and predicted).
    let mut error_groups: Vec<(i64, String, String)> = Vec::new(); // (idx, case, role)
    let mut seen_error = std::collections::HashSet::new();
    for r in results {
        if r.pred_letter_idx != r.letter_idx {
            let true_key = (r.letter_idx, r.case_str.clone());
            if seen_error.insert(true_key.clone()) {
                error_groups.push((r.letter_idx, r.case_str.clone(), "true".to_string()));
            }
            // Predicted letter — same case (model predicted wrong letter but same case context).
            let pred_key = (r.pred_letter_idx, r.case_str.clone());
            if seen_error.insert(pred_key.clone()) {
                error_groups.push((r.pred_letter_idx, r.case_str.clone(), "pred".to_string()));
            }
        }
    }
    if !error_groups.is_empty() {
        html.push_str("<div class=\"ebar\">\n");
        html.push_str("<span class=\"ebar-label\">Errors</span>");
        for (idx, cas, role) in &error_groups {
            let ch = LETTERS[*idx as usize];
            let display = if cas == "upper" {
                ch.to_uppercase().to_string()
            } else {
                ch.to_lowercase().to_string()
            };
            let hue = (*idx as f64 / 26.0 * 360.0) as u32;
            let border = if role == "true" { "#f44336" } else { "#ff9800" };
            let label = if role == "true" { "was" } else { "predicted" };
            let _ = write!(html,
                "<button class=\"lbtn ebtn\" data-idx=\"{}\" data-cas=\"{}\" title=\"{} ({}) — {}\" style=\"background:hsl({},70%,25%);color:hsl({},70%,70%);border:2px solid {}\">{}</button>",
                idx, cas, display, cas, label, hue, hue, border, display);
        }
        // Error details text.
        html.push_str("<span class=\"ebar-details\">");
        for r in results {
            if r.pred_letter_idx == r.letter_idx { continue; }
            let pred_ch = LETTERS[r.pred_letter_idx as usize];
            let _ = write!(html, "<span>{} ({}) predicted as {} — {}</span>",
                r.letter, r.case_str, pred_ch, r.font);
        }
        html.push_str("</span>");
        html.push_str("</div>\n");
    }

    // --- Letter bar: two rows ---
    html.push_str("<div class=\"lbar\" id=\"lbar\">\n");
    html.push_str("<div class=\"lrow\"><span class=\"lbl\">Upper</span>");
    for i in 0..26u32 {
        let ch = (b'A' + i as u8) as char;
        let hue = (i as f64 / 26.0 * 360.0) as u32;
        let _ = write!(html,
            "<button class=\"lbtn\" data-idx=\"{}\" data-cas=\"upper\" style=\"background:hsl({},70%,25%);color:hsl({},70%,70%)\">{}</button>",
            i, hue, hue, ch);
    }
    html.push_str("</div>\n<div class=\"lrow\"><span class=\"lbl\">Lower</span>");
    for i in 0..26u32 {
        let ch = (b'a' + i as u8) as char;
        let hue = (i as f64 / 26.0 * 360.0) as u32;
        let _ = write!(html,
            "<button class=\"lbtn\" data-idx=\"{}\" data-cas=\"lower\" style=\"background:hsl({},70%,25%);color:hsl({},70%,70%)\">{}</button>",
            i, hue, hue, ch);
    }
    html.push_str("</div>\n</div>\n");

    // --- Canvas + controls ---
    html.push_str("<canvas id=\"c\"></canvas>\n");
    html.push_str("<div class=\"controls\"><span class=\"sel\" id=\"sel\">All letters</span><span style=\"color:#888\">Click dot / letter button / table row to select | Esc or click background to reset</span></div>\n");
    html.push_str("<div class=\"tooltip\" id=\"tip\"></div>\n");

    // Emit data as JSON: [x, y, letter_idx, case, letter, font].
    html.push_str("<script>\nconst DATA=[\n");
    for (x, y, idx, case, letter, font) in &points {
        let _ = writeln!(html, "[{:.4},{:.4},{},\"{}\",\"{}\",\"{}\"],", x, y, idx, case, letter, font);
    }
    html.push_str("];\n");

    // Emit silhouette scores as lookup: "idx_case" -> score.
    html.push_str("const SIL={");
    for (key, &score) in &sil_scores {
        let _ = write!(html, "\"{}\":{:.4},", key, score);
    }
    html.push_str("};\n");
    let _ = writeln!(html, "function getSil(d){{return SIL[d[2]+'_'+d[3]]||0;}}");

    // Color palette — 26 distinct hues.
    html.push_str("const COLORS=[");
    for i in 0..26 {
        let hue = (i as f64 / 26.0 * 360.0) as u32;
        let _ = write!(html, "\"hsl({},70%,60%)\",", hue);
    }
    html.push_str("];\n");

    // Render with selection support.
    html.push_str(r#"
const canvas=document.getElementById('c');
const ctx=canvas.getContext('2d');
const tip=document.getElementById('tip');
const sel=document.getElementById('sel');
const PAD=40;
let W,H;

function sizeCanvas(){
  const dpr=window.devicePixelRatio||1;
  const cssW=canvas.clientWidth;
  const cssH=Math.round(cssW*0.58);
  canvas.width=cssW*dpr;
  canvas.height=cssH*dpr;
  canvas.style.height=cssH+'px';
  ctx.setTransform(dpr,0,0,dpr,0,0);
  W=cssW;H=cssH;
}
sizeCanvas();

let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
for(const d of DATA){minX=Math.min(minX,d[0]);maxX=Math.max(maxX,d[0]);minY=Math.min(minY,d[1]);maxY=Math.max(maxY,d[1]);}
const rangeX=maxX-minX||1,rangeY=maxY-minY||1;
function tx(x){return PAD+(x-minX)/rangeX*(W-2*PAD)}
function ty(y){return H-PAD-(y-minY)/rangeY*(H-2*PAD)}

window.addEventListener('resize',()=>{sizeCanvas();draw();});

// Selection state: {idx, cas} or null.
let selected=null;

function isSelected(d){
  if(!selected) return true;
  return d[2]===selected.idx && d[3]===selected.cas;
}

// Find the display letter for a given index+case.
function letterFor(idx,cas){
  const d=DATA.find(d=>d[2]===idx&&d[3]===cas);
  return d?d[4]:'?';
}

function selectGroup(idx,cas){
  selected={idx,cas};
  const n=DATA.filter(d=>isSelected(d)).length;
  const s=SIL[idx+'_'+cas]||0;
  const letter=letterFor(idx,cas);
  sel.innerHTML=letter+' ('+cas+') &mdash; '+n+' samples &mdash; silhouette: <b>'+s.toFixed(3)+'</b>';
  sel.style.color=COLORS[idx];
  updateActiveBtn();
  draw();
}

function clearSelection(){
  selected=null;
  sel.textContent='All letters';
  sel.style.color='#fff';
  updateActiveBtn();
  draw();
}

function updateActiveBtn(){
  document.querySelectorAll('.lbtn').forEach(b=>{
    const bi=parseInt(b.dataset.idx),bc=b.dataset.cas;
    b.classList.toggle('active',selected&&selected.idx===bi&&selected.cas===bc);
  });
}

function draw(){
  ctx.clearRect(0,0,W,H);
  // Draw dimmed dots first, then highlighted on top.
  for(const pass of [0,1]){
    for(const d of DATA){
      const hi=isSelected(d);
      if(pass===0 && hi) continue;
      if(pass===1 && !hi) continue;
      const x=tx(d[0]),y=ty(d[1]);
      const r=hi?(d[3]==='lower'?5:6):(d[3]==='lower'?3:4);
      ctx.beginPath();
      ctx.arc(x,y,r,0,Math.PI*2);
      ctx.fillStyle=COLORS[d[2]];
      ctx.globalAlpha=hi?1.0:selected?0.12:(d[3]==='lower'?0.6:0.85);
      ctx.fill();
      if(hi&&selected){ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.stroke();}
      else if(d[3]==='upper'&&!selected){ctx.strokeStyle='#fff';ctx.lineWidth=0.5;ctx.stroke();}
    }
  }
  ctx.globalAlpha=1;
  // Draw font labels on highlighted dots.
  if(selected){
    ctx.font='bold 11px system-ui';ctx.textAlign='center';ctx.textBaseline='bottom';
    ctx.fillStyle='#fff';
    for(const d of DATA){
      if(!isSelected(d)) continue;
      ctx.fillText(d[5],tx(d[0]),ty(d[1])-8);
    }
  }
}
draw();

function hitTest(mx,my){
  let best=null,bestD=25;
  for(const d of DATA){
    const dx=tx(d[0])-mx,dy=ty(d[1])-my;
    const dist=dx*dx+dy*dy;
    if(dist<bestD){bestD=dist;best=d;}
  }
  return best;
}

canvas.addEventListener('mousemove',e=>{
  const rect=canvas.getBoundingClientRect();
  const mx=e.clientX-rect.left,my=e.clientY-rect.top;
  const best=hitTest(mx,my);
  if(best){
    tip.style.display='block';
    tip.style.left=(e.pageX+12)+'px';
    tip.style.top=(e.pageY-20)+'px';
    tip.innerHTML=best[4]+' ('+best[3]+') <b>'+best[5]+'</b><br>PC1='+best[0].toFixed(2)+' PC2='+best[1].toFixed(2)+'<br>Silhouette: '+getSil(best).toFixed(3);
  }else{tip.style.display='none';}
});

canvas.addEventListener('click',e=>{
  const rect=canvas.getBoundingClientRect();
  const mx=e.clientX-rect.left,my=e.clientY-rect.top;
  const best=hitTest(mx,my);
  if(best){selectGroup(best[2],best[3]);}
  else{clearSelection();}
});

document.addEventListener('keydown',e=>{
  if(e.key==='Escape'&&selected) clearSelection();
});

// Letter bar clicks.
document.getElementById('lbar').addEventListener('click',e=>{
  const btn=e.target.closest('.lbtn');
  if(!btn) return;
  const idx=parseInt(btn.dataset.idx),cas=btn.dataset.cas;
  if(selected&&selected.idx===idx&&selected.cas===cas) clearSelection();
  else selectGroup(idx,cas);
});

// Letter bar hover tooltips.
document.getElementById('lbar').addEventListener('mouseover',e=>{
  const btn=e.target.closest('.lbtn');
  if(!btn) return;
  const idx=parseInt(btn.dataset.idx),cas=btn.dataset.cas;
  const s=SIL[idx+'_'+cas]||0;
  btn.title=letterFor(idx,cas)+' ('+cas+') silhouette: '+s.toFixed(3);
});

// Summary table row clicks.
document.querySelectorAll('.scard tr[data-idx]').forEach(tr=>{
  tr.addEventListener('click',()=>{
    const idx=parseInt(tr.dataset.idx),cas=tr.dataset.cas;
    if(selected&&selected.idx===idx&&selected.cas===cas) clearSelection();
    else selectGroup(idx,cas);
  });
});

// Error bar clicks (if present).
const ebar=document.querySelector('.ebar');
if(ebar){
  ebar.addEventListener('click',e=>{
    const btn=e.target.closest('.lbtn');
    if(!btn) return;
    const idx=parseInt(btn.dataset.idx),cas=btn.dataset.cas;
    if(selected&&selected.idx===idx&&selected.cas===cas) clearSelection();
    else selectGroup(idx,cas);
  });
}
"#);
    html.push_str("</script>\n</body></html>");
    html
}

/// Simple power iteration to find the dominant eigenvector of X^T X.
fn power_iteration(data: &[Vec<f64>], dim: usize, iters: usize) -> Vec<f64> {
    let mut v = vec![0.0f64; dim];
    // Initialize with first row.
    if !data.is_empty() {
        v.clone_from(&data[0]);
    }
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 { for x in &mut v { *x /= norm; } }

    for _ in 0..iters {
        // w = X^T (X v)
        let mut w = vec![0.0f64; dim];
        for row in data {
            let dot: f64 = row.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum();
            for (wi, &ri) in w.iter_mut().zip(row.iter()) {
                *wi += dot * ri;
            }
        }
        let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 { for x in &mut w { *x /= norm; } }
        v = w;
    }
    v
}

/// Encode grayscale f32 pixels [0,1] to PNG bytes.
fn pixels_to_png(data: &[f32], w: usize, h: usize) -> Vec<u8> {
    let mut buf = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut buf, w as u32, h as u32);
        encoder.set_color(png::ColorType::Grayscale);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = match encoder.write_header() {
            Ok(w) => w,
            Err(_) => return Vec::new(),
        };
        let bytes: Vec<u8> = data.iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8)
            .collect();
        let _ = writer.write_image_data(&bytes);
    }
    buf
}
