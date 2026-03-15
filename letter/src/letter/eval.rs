//! Evaluation module — loads a trained model, runs inference on test data,
//! generates accuracy reports, and produces an attention atlas HTML.

use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::Path;

use flodl::autograd::{Variable, NoGradGuard};
use flodl::tensor::{cuda_available, Device, Result, Tensor, TensorError};
use serde::Deserialize;

use super::data::load_gray_png;
use super::model::LetterModel;
use super::train::LetterConfig;

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
    test_data_dir: &str,
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

    eprintln!("Config: {} scan + {} read, latent_dim={}", cfg.n_scan, cfg.n_read, cfg.latent_dim);
    eprintln!("Model:  {}", model_path.display());

    // Create model and load checkpoint.
    let model = LetterModel::new(
        cfg.n_classes, cfg.n_scan, cfg.n_read, cfg.patch_size, cfg.scan_patch_w,
        cfg.n_scales, cfg.latent_dim,
    )?;

    let device = if cuda_available() {
        eprintln!("Using CUDA");
        model.graph.set_device(Device::CUDA);
        Device::CUDA
    } else {
        eprintln!("Using CPU");
        Device::CPU
    };

    let report = model.graph.load_checkpoint(&model_path.to_string_lossy())?;
    eprintln!("Checkpoint: {} loaded, {} skipped, {} missing",
        report.loaded.len(), report.skipped.len(), report.missing.len());
    model.set_training(false);

    // Load test data.
    let samples = load_test_dataset(test_data_dir)?;

    // Run inference under NoGradGuard.
    let _guard = NoGradGuard::new();
    let mut results: Vec<SampleResult> = Vec::with_capacity(samples.len());

    for (i, sample) in samples.iter().enumerate() {
        let img = sample.image.unsqueeze(0)?.to_device(device)?; // [1, 1, H, W]
        let case = Tensor::from_f32(&[sample.case_label], &[1, 1], device)?;

        let img_var = Variable::new(img, false);
        let case_var = Variable::new(case, false);

        let result = model.forward(&img_var, &case_var)?;

        // Predictions.
        let pred_letter = result.letter_logits.data().argmax(1, false)?
            .to_i64_vec()?[0];
        let pred_case = result.case_logits.data().argmax(1, false)?
            .to_i64_vec()?[0];

        // Reconstruction MSE.
        let recon_data = result.recon.data().to_device(Device::CPU)?;
        let input_data = sample.image.unsqueeze(0)?;
        let diff = recon_data.sub(&input_data)?;
        let mse = diff.mul(&diff)?.mean()?.item()?;

        // Fixation locations.
        let scan_locs = extract_locs(&result.scan_locations)?;
        let read_locs = extract_locs(&result.read_locations)?;

        // Reconstruction pixels for atlas.
        let recon_pixels = result.recon.data().to_device(Device::CPU)?
            .squeeze(0)?.squeeze(0)?.to_f32_vec()?;

        // Read original PNG bytes for atlas embedding.
        let img_path = resolve_test_path(Path::new(test_data_dir), &format!(
            "img_{}_{}.png",
            &sample.letter,
            if sample.font == "unknown" { "default" } else { &sample.font }
        ));
        let png_bytes = fs::read(&img_path).unwrap_or_default();

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

    let atlas = generate_atlas(&results);
    fs::write(format!("{}/letter_atlas.html", save_path), &atlas)
        .map_err(|e| TensorError::new(&format!("write atlas: {e}")))?;

    eprintln!("Saved: {}/results.md", save_path);
    eprintln!("Saved: {}/eval.json", save_path);
    eprintln!("Saved: {}/letter_atlas.html", save_path);

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
    let _ = writeln!(s, "Letter: {}/{} ({:.1}%)", letter_correct, total,
        letter_correct as f64 / total as f64 * 100.0);
    let _ = writeln!(s, "Case:   {}/{} ({:.1}%)", case_correct, total,
        case_correct as f64 / total as f64 * 100.0);
    let _ = writeln!(s, "Avg MSE:      {:.4}", avg_mse);

    // Per-font breakdown.
    let mut fonts: Vec<String> = results.iter().map(|r| r.font.clone()).collect();
    fonts.sort();
    fonts.dedup();

    let _ = writeln!(s, "Per-font:");
    for font in &fonts {
        let font_results: Vec<&SampleResult> = results.iter()
            .filter(|r| &r.font == font)
            .collect();
        let n = font_results.len();
        let fl = font_results.iter().filter(|r| r.pred_letter_idx == r.letter_idx).count();
        let fc = font_results.iter().filter(|r| r.case_correct).count();
        let _ = writeln!(s, "  {:25}: Letter {:.1}%  Case {:.1}%  ({})",
            font, fl as f64 / n as f64 * 100.0, fc as f64 / n as f64 * 100.0, n);
    }

    // Errors (if any).
    let errors: Vec<&SampleResult> = results.iter()
        .filter(|r| r.pred_letter_idx != r.letter_idx || !r.case_correct)
        .collect();
    if errors.is_empty() {
        let _ = writeln!(s, "\nNo errors.");
    } else {
        let _ = writeln!(s, "\nErrors ({}):", errors.len());
        for r in &errors {
            let pred = idx_to_letter(r.pred_letter_idx, &r.case_str);
            let _ = writeln!(s, "  {} ({}) → predicted {} (font: {})",
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
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
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

/// Generate self-contained HTML attention atlas.
fn generate_atlas(results: &[SampleResult]) -> String {

    let total = results.len();
    let letter_correct = results.iter()
        .filter(|r| r.pred_letter_idx == r.letter_idx).count();
    let case_correct = results.iter().filter(|r| r.case_correct).count();

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

    let mut html = String::with_capacity(1024 * 1024);

    // Header.
    html.push_str("<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\">\n");
    html.push_str("<title>Letter Attention Atlas</title>\n");
    html.push_str("<style>\n");
    html.push_str("body{font-family:system-ui,-apple-system,sans-serif;background:#1a1a2e;color:#e0e0e0;margin:20px}\n");
    html.push_str("h1{color:#fff;margin-bottom:4px} h2{color:#ccc;margin:24px 0 8px;border-bottom:1px solid #333;padding-bottom:4px}\n");
    html.push_str(".summary{color:#aaa;margin-bottom:20px}\n");
    html.push_str(".letter-row{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}\n");
    html.push_str(".sample{display:inline-block;text-align:center;background:#252540;border-radius:6px;padding:6px}\n");
    html.push_str(".sample.error{border:2px solid #e74c3c}\n");
    html.push_str(".label{font-size:11px;color:#888;margin-top:2px}\n");
    html.push_str(".pred{font-size:11px;font-weight:bold}\n");
    html.push_str(".pred.correct{color:#2ecc71} .pred.wrong{color:#e74c3c}\n");
    html.push_str(".pair{display:flex;gap:2px}\n");
    html.push_str(".legend{margin:16px 0;padding:10px;background:#252540;border-radius:6px;display:inline-block}\n");
    html.push_str(".legend span{margin-right:16px}\n");
    html.push_str("</style></head><body>\n");

    let _ = write!(html, "<h1>Letter Attention Atlas</h1>\n");
    let _ = write!(html, "<div class=\"summary\">Letter: {}/{} ({:.1}%) &middot; Case: {}/{} ({:.1}%)</div>\n",
        letter_correct, total, letter_correct as f64 / total as f64 * 100.0,
        case_correct, total, case_correct as f64 / total as f64 * 100.0);

    // Legend.
    html.push_str("<div class=\"legend\">\n");
    html.push_str("<span><svg width=\"12\" height=\"12\"><circle cx=\"6\" cy=\"6\" r=\"5\" fill=\"#3498db\"/></svg> Scan</span>\n");
    html.push_str("<span><svg width=\"12\" height=\"12\"><circle cx=\"6\" cy=\"6\" r=\"5\" fill=\"#e74c3c\"/></svg> Read (numbered)</span>\n");
    html.push_str("<span>Left: input + fixations &middot; Right: reconstruction</span>\n");
    html.push_str("</div>\n");

    // Per-letter sections.
    for (letter, samples) in &by_letter {
        let _ = write!(html, "<h2>{}</h2>\n<div class=\"letter-row\">\n", letter);

        for r in samples {
            let is_correct = r.pred_letter_idx == r.letter_idx && r.case_correct;
            let pred = idx_to_letter(r.pred_letter_idx, &r.case_str);
            let error_class = if is_correct { "" } else { " error" };

            let _ = write!(html, "<div class=\"sample{}\">\n<div class=\"pair\">\n", error_class);

            // Input image with fixation overlay (SVG).
            let _ = write!(html, "<svg width=\"128\" height=\"128\" viewBox=\"0 0 128 128\">\n");
            if !r.image_png.is_empty() {
                let img_b64 = base64_encode(&r.image_png);
                let _ = write!(html, "<image href=\"data:image/png;base64,{}\" width=\"128\" height=\"128\"/>\n", img_b64);
            }
            // Scan fixations (blue).
            for (i, loc) in r.scan_locs.iter().enumerate() {
                let px = (loc[0] + 1.0) / 2.0 * 128.0;
                let py = (loc[1] + 1.0) / 2.0 * 128.0;
                let _ = write!(html, "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"5\" fill=\"#3498db\" opacity=\"0.8\"/>\n", px, py);
                let _ = write!(html, "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"#fff\" font-size=\"7\" text-anchor=\"middle\" dominant-baseline=\"central\">S{}</text>\n",
                    px, py, i + 1);
            }
            // Read fixations (red, numbered).
            for (i, loc) in r.read_locs.iter().enumerate() {
                let px = (loc[0] + 1.0) / 2.0 * 128.0;
                let py = (loc[1] + 1.0) / 2.0 * 128.0;
                let _ = write!(html, "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"4\" fill=\"#e74c3c\" opacity=\"0.8\"/>\n", px, py);
                let _ = write!(html, "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"#fff\" font-size=\"7\" text-anchor=\"middle\" dominant-baseline=\"central\">{}</text>\n",
                    px, py, i + 1);
            }
            html.push_str("</svg>\n");

            // Reconstruction image (generated from pixel data).
            if r.recon_data.len() == 128 * 128 {
                let recon_png = pixels_to_png(&r.recon_data, 128, 128);
                let recon_b64 = base64_encode(&recon_png);
                let _ = write!(html, "<img src=\"data:image/png;base64,{}\" width=\"128\" height=\"128\">\n", recon_b64);
            }

            html.push_str("</div>\n"); // .pair

            // Labels.
            let _ = write!(html, "<div class=\"label\">{}</div>\n", r.font);
            let correct_class = if is_correct { "correct" } else { "wrong" };
            let _ = write!(html, "<div class=\"pred {}\">→ {}</div>\n", correct_class, pred);
            html.push_str("</div>\n"); // .sample
        }
        html.push_str("</div>\n"); // .letter-row
    }

    html.push_str("</body></html>");
    html
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
