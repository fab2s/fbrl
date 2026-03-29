//! In-memory font-based letter dataset generator.
//!
//! Loads TTF/OTF fonts, rasterizes letters with 0-2 random neighbors,
//! applies gaussian blur variants. Returns a `LetterDataset` directly
//! in RAM — no disk I/O needed for training.
//!
//! Images are white-on-black (ink=1.0, background=0.0).

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use fontdue::{Font, FontSettings};
use flodl::tensor::{Device, Result, Tensor, TensorError};
use serde::{Serialize, Deserialize};

use super::data::{LetterDataset, LetterSample};

/// Generator configuration — loaded from JSON.
#[derive(Serialize, Deserialize, Clone)]
pub struct GenConfig {
    /// Glob pattern for TTF/OTF font files.
    pub fonts: String,
    /// Characters to use (must include both cases).
    pub charset: String,
    /// Image height in pixels.
    #[serde(default = "default_height")]
    pub image_height: usize,
    /// Image width in pixels.
    #[serde(default = "default_width")]
    pub image_width: usize,
    /// Minimum letters per image (1 = alone, 2 = one neighbor, 3 = both).
    #[serde(default = "default_min_letters")]
    pub min_letters: usize,
    /// Maximum letters per image.
    #[serde(default = "default_max_letters")]
    pub max_letters: usize,
    /// Number of base renderings to generate.
    #[serde(default = "default_samples")]
    pub samples: usize,
    /// Fraction of image height used by text.
    #[serde(default = "default_fill")]
    pub fill_ratio: f64,
    /// RNG seed for reproducibility.
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// Total copies per base rendering (1 = clean only, 3 = 1 clean + 2 blurred).
    #[serde(default = "default_blur_variants")]
    pub blur_variants: usize,
    /// Maximum gaussian blur sigma for blurred variants.
    #[serde(default = "default_blur_sigma_max")]
    pub blur_sigma_max: f64,
    /// Additive gaussian noise level (std dev). 0.0 = no noise.
    #[serde(default = "default_noise_level")]
    pub noise_level: f64,
}

fn default_height() -> usize { 128 }
fn default_width() -> usize { 256 }
fn default_min_letters() -> usize { 1 }
fn default_max_letters() -> usize { 3 }
fn default_samples() -> usize { 20000 }
fn default_fill() -> f64 { 0.80 }
fn default_seed() -> u64 { 0xF0D1_CAFE }
fn default_blur_variants() -> usize { 1 }
fn default_blur_sigma_max() -> f64 { 1.5 }
fn default_noise_level() -> f64 { 0.1 }

struct LoadedFont {
    font: Font,
    px_size: f32,
    baseline_y: i32,
    name: String,
}

/// Generate a `LetterDataset` in memory from font files.
///
/// Each base rendering has the target letter centered with 0-2 random neighbors.
/// Blur variants multiply the dataset: each base produces `blur_variants` copies
/// (one always clean, the rest with random gaussian blur).
///
/// Total samples = `cfg.samples * cfg.blur_variants`.
pub fn generate_letter_dataset(cfg: &GenConfig) -> Result<LetterDataset> {
    let fonts = load_fonts(cfg)?;
    if fonts.is_empty() {
        return Err(TensorError::new("no fonts loaded"));
    }

    let chars: Vec<char> = cfg.charset.chars().collect();
    if chars.is_empty() {
        return Err(TensorError::new("empty charset"));
    }

    let blur_n = cfg.blur_variants.max(1);

    let spacing = 0.5;
    let center_norm = 0.0;
    let left_norm = -spacing;
    let right_norm = spacing;

    let h = cfg.image_height;
    let w = cfg.image_width;
    let mut rng = cfg.seed;

    // Phase 1: generate all samples, track identity for partner resolution.
    struct RawSample {
        image: Tensor,
        clean: Tensor,
        letter_idx: i64,
        case_label: f32,
        font_name: String,
    }

    // Balanced distribution: every (font, char) combo gets equal representation.
    // Center letter is perfectly balanced; neighbors are random noise.
    let n_combos = fonts.len() * chars.len();
    let per_combo = cfg.samples / n_combos;
    let remainder = cfg.samples % n_combos;
    let total_base = per_combo * n_combos + remainder;
    let total = total_base * blur_n;

    eprintln!("Letter generator: {} fonts × {} chars = {} combos, {} per combo (+{} extra) = {} base × {} blur = {} samples",
        fonts.len(), chars.len(), n_combos, per_combo, remainder,
        total_base, blur_n, total);

    let mut raw_samples = Vec::with_capacity(total);
    let mut clean_by_id: HashMap<(i64, bool, String), Tensor> = HashMap::new();

    let mut combo_idx = 0usize;
    for lf in &fonts {
        for &target_ch in &chars {
            let letter_idx = char_to_idx(target_ch);
            let case_label = if target_ch.is_lowercase() { 1.0f32 } else { 0.0f32 };
            let repeats = per_combo + if combo_idx < remainder { 1 } else { 0 };
            combo_idx += 1;

            for _ in 0..repeats {
                // Random neighbor count.
                let n = if cfg.min_letters == cfg.max_letters {
                    cfg.min_letters
                } else {
                    cfg.min_letters + rng_range(&mut rng, cfg.max_letters - cfg.min_letters + 1)
                };

                let mut pixels = vec![0.0f32; h * w];
                render_glyph_at(&lf.font, lf.px_size, lf.baseline_y, target_ch,
                                center_norm, &mut pixels, h, w);
                if n >= 2 {
                    let neighbor_ch = chars[rng_range(&mut rng, chars.len())];
                    if n == 2 {
                        let pos = if rng_range(&mut rng, 2) == 0 { left_norm } else { right_norm };
                        render_glyph_at(&lf.font, lf.px_size, lf.baseline_y, neighbor_ch,
                                        pos, &mut pixels, h, w);
                    } else {
                        render_glyph_at(&lf.font, lf.px_size, lf.baseline_y, neighbor_ch,
                                        left_norm, &mut pixels, h, w);
                        let right_ch = chars[rng_range(&mut rng, chars.len())];
                        render_glyph_at(&lf.font, lf.px_size, lf.baseline_y, right_ch,
                                        right_norm, &mut pixels, h, w);
                    }
                }

                let clean = Tensor::from_f32(&pixels, &[1, h as i64, w as i64], Device::CPU)?;
                let is_lower = case_label > 0.5;
                clean_by_id.entry((letter_idx, is_lower, lf.name.clone()))
                    .or_insert_with(|| clean.clone());

                // Generate blur + noise variants.
                for v in 0..blur_n {
                    let image = if v == 0 {
                        clean.clone()
                    } else {
                        let mut degraded = if cfg.blur_sigma_max > 0.0 {
                            let sigma = rng_f64(&mut rng) * cfg.blur_sigma_max;
                            if sigma < 0.1 { pixels.clone() }
                            else { gaussian_blur_pixels(&pixels, h, w, sigma) }
                        } else {
                            pixels.clone()
                        };
                        if cfg.noise_level > 0.0 {
                            let nl = cfg.noise_level as f32;
                            for p in &mut degraded {
                                *p = (*p + rng_gaussian(&mut rng) * nl).clamp(0.0, 1.0);
                            }
                        }
                        Tensor::from_f32(&degraded, &[1, h as i64, w as i64], Device::CPU)?
                    };

                    raw_samples.push(RawSample {
                        image,
                        clean: clean.clone(),
                        letter_idx,
                        case_label,
                        font_name: lf.name.clone(),
                    });
                }
            }
        }
    }

    // Phase 2: resolve partners (opposite case, same letter, same font).
    let mut has_partners = true;
    let mut samples = Vec::with_capacity(raw_samples.len());
    for raw in raw_samples {
        let is_lower = raw.case_label > 0.5;
        let partner_clean = if let Some(p) = clean_by_id.get(&(raw.letter_idx, !is_lower, raw.font_name.clone())) {
            p.clone()
        } else {
            has_partners = false;
            Tensor::zeros(&raw.clean.shape(), Default::default())?
        };
        samples.push(LetterSample {
            image: raw.image,
            clean: raw.clean,
            letter_idx: raw.letter_idx,
            case_label: raw.case_label,
            font: raw.font_name.clone(),
            partner_clean,
        });
    }

    if !has_partners {
        eprintln!("Note: some letters missing opposite-case partners");
    }
    eprintln!("Generated {} letter samples in memory ({} with partners)",
        samples.len(), if has_partners { "all" } else { "some" });

    Ok(LetterDataset { samples, has_partners })
}

/// Save generated images to disk for debugging.
pub fn save_letter_dataset(
    ds: &LetterDataset, h: usize, w: usize, dir: &str,
) -> Result<()> {
    let dir_path = Path::new(dir);
    fs::create_dir_all(dir_path)
        .map_err(|e| TensorError::new(&format!("create dir: {e}")))?;

    let mut meta: HashMap<String, serde_json::Value> = HashMap::new();

    for (i, sample) in ds.samples.iter().enumerate() {
        let img_name = format!("img_{:05}.png", i);
        let img_path = dir_path.join(&img_name);

        let pixels = sample.image.to_f32_vec()?;
        save_png(&pixels, h, w, &img_path)?;

        let letter = ((sample.letter_idx as u8 + b'A') as char).to_string();
        let case = if sample.case_label > 0.5 { "lower" } else { "upper" };
        meta.insert(img_name.clone(), serde_json::json!({
            "image": img_name,
            "letter": letter,
            "case": case,
            "font": sample.font,
        }));
    }

    fs::write(
        dir_path.join("metadata.json"),
        serde_json::to_string_pretty(&meta).unwrap_or_default(),
    ).map_err(|e| TensorError::new(&format!("write metadata: {e}")))?;

    eprintln!("Saved {} samples to {}", ds.samples.len(), dir);
    Ok(())
}

// ── Gaussian blur on raw pixels ─────────────────────────────────────

fn gaussian_blur_pixels(pixels: &[f32], h: usize, w: usize, sigma: f64) -> Vec<f32> {
    let radius = (3.0 * sigma).ceil() as usize;
    let kernel = gaussian_kernel(sigma, radius);

    // Horizontal pass.
    let mut temp = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for (k, &kv) in kernel.iter().enumerate() {
                let sx = (x as isize + k as isize - radius as isize)
                    .clamp(0, w as isize - 1) as usize;
                sum += pixels[y * w + sx] * kv;
            }
            temp[y * w + x] = sum;
        }
    }

    // Vertical pass.
    let mut out = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for (k, &kv) in kernel.iter().enumerate() {
                let sy = (y as isize + k as isize - radius as isize)
                    .clamp(0, h as isize - 1) as usize;
                sum += temp[sy * w + x] * kv;
            }
            out[y * w + x] = sum;
        }
    }
    out
}

fn gaussian_kernel(sigma: f64, radius: usize) -> Vec<f32> {
    let size = 2 * radius + 1;
    let mut kernel = Vec::with_capacity(size);
    let mut sum = 0.0f64;
    for i in 0..size {
        let x = i as f64 - radius as f64;
        let v = (-x * x / (2.0 * sigma * sigma)).exp();
        kernel.push(v as f32);
        sum += v;
    }
    let norm = 1.0 / sum as f32;
    for v in &mut kernel { *v *= norm; }
    kernel
}

// ── Font loading ────────────────────────────────────────────────────

fn load_fonts(cfg: &GenConfig) -> Result<Vec<LoadedFont>> {
    let paths: Vec<_> = glob::glob(&cfg.fonts)
        .map_err(|e| TensorError::new(&format!("invalid fonts glob: {e}")))?
        .filter_map(|r| r.ok())
        .collect();

    if paths.is_empty() {
        return Err(TensorError::new(&format!(
            "no fonts found matching '{}'", cfg.fonts
        )));
    }

    let mut fonts = Vec::with_capacity(paths.len());
    for path in &paths {
        let data = fs::read(path)
            .map_err(|e| TensorError::new(&format!("read {}: {e}", path.display())))?;

        let font = match Font::from_bytes(data, FontSettings::default()) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("warning: skip {}: {}", path.display(), e);
                continue;
            }
        };

        // Compute px_size so line height fills fill_ratio of image.
        let target_px = (cfg.image_height as f64 * cfg.fill_ratio) as f32;
        let mut px_size = target_px;
        for _ in 0..5 {
            if let Some(lm) = font.horizontal_line_metrics(px_size) {
                let actual_height = lm.ascent - lm.descent;
                if actual_height > 0.1 {
                    px_size = px_size * target_px / actual_height;
                }
            }
        }

        let baseline_y = if let Some(lm) = font.horizontal_line_metrics(px_size) {
            let total = lm.ascent - lm.descent;
            let top_margin = ((cfg.image_height as f32 - total) / 2.0).max(0.0);
            (top_margin + lm.ascent) as i32
        } else {
            (cfg.image_height as f32 * 0.75) as i32
        };

        let name = path.file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".into());

        fonts.push(LoadedFont { font, px_size, baseline_y, name });
    }

    eprintln!("Loaded {} fonts from '{}'", fonts.len(), cfg.fonts);
    Ok(fonts)
}

// ── Glyph rendering ─────────────────────────────────────────────────

fn render_glyph_at(
    font: &Font, px_size: f32, baseline_y: i32, ch: char,
    norm_x: f64, pixels: &mut [f32], img_h: usize, img_w: usize,
) {
    let (metrics, bitmap) = font.rasterize(ch, px_size);
    if metrics.width == 0 || metrics.height == 0 { return; }

    let center_x = ((norm_x + 1.0) / 2.0 * img_w as f64) as i32;
    let glyph_x = center_x - metrics.width as i32 / 2 + metrics.xmin;
    let glyph_y = baseline_y - metrics.height as i32 - metrics.ymin;

    for gy in 0..metrics.height {
        for gx in 0..metrics.width {
            let px = glyph_x + gx as i32;
            let py = glyph_y + gy as i32;
            if px < 0 || py < 0 || px >= img_w as i32 || py >= img_h as i32 {
                continue;
            }
            let coverage = bitmap[gy * metrics.width + gx] as f32 / 255.0;
            let idx = py as usize * img_w + px as usize;
            pixels[idx] = (pixels[idx] + coverage).min(1.0);
        }
    }
}

fn char_to_idx(ch: char) -> i64 {
    let upper = ch.to_ascii_uppercase();
    if upper.is_ascii_uppercase() {
        (upper as u8 - b'A') as i64
    } else {
        -1
    }
}

fn save_png(pixels: &[f32], h: usize, w: usize, path: &Path) -> Result<()> {
    let bytes: Vec<u8> = pixels.iter()
        .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect();
    let file = fs::File::create(path)
        .map_err(|e| TensorError::new(&format!("create {}: {e}", path.display())))?;
    let mut encoder = png::Encoder::new(file, w as u32, h as u32);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()
        .map_err(|e| TensorError::new(&format!("png header: {e}")))?;
    writer.write_image_data(&bytes)
        .map_err(|e| TensorError::new(&format!("png write: {e}")))?;
    Ok(())
}

// ── RNG ─────────────────────────────────────────────────────────────

#[inline]
fn rng_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

#[inline]
fn rng_range(state: &mut u64, n: usize) -> usize {
    (rng_next(state) >> 33) as usize % n
}

#[inline]
fn rng_f64(state: &mut u64) -> f64 {
    (rng_next(state) >> 11) as f64 / (1u64 << 53) as f64
}

#[inline]
fn rng_gaussian(state: &mut u64) -> f32 {
    let u1 = rng_f64(state).max(1e-10);
    let u2 = rng_f64(state);
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
}
