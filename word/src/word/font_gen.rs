//! In-memory font-based dataset generator.
//!
//! Loads TTF/OTF fonts and a word list, rasterizes words with proportional
//! glyph spacing. Returns a `WordDataset` directly in RAM.
//!
//! Images are white-on-black (ink=1.0, background=0.0).

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use fontdue::{Font, FontSettings};
use flodl::tensor::{Device, Result, Tensor, TensorError};
use serde::{Serialize, Deserialize};

use fbrl::letter::{LetterDataset, LetterSample};
use super::data::{WordDataset, WordSample};

/// Generator configuration — loaded from JSON.
#[derive(Serialize, Deserialize, Clone)]
pub struct GenConfig {
    /// Glob pattern for TTF/OTF font files.
    pub fonts: String,
    /// Characters to use (for letter-mode and random fallback).
    pub charset: String,
    /// Image height in pixels.
    #[serde(default = "default_height")]
    pub image_height: usize,
    /// Image width in pixels.
    #[serde(default = "default_width")]
    pub image_width: usize,
    /// Minimum letters per image.
    #[serde(default = "default_min_letters")]
    pub min_letters: usize,
    /// Maximum letters per image.
    #[serde(default = "default_max_letters")]
    pub max_letters: usize,
    /// Number of samples to generate.
    #[serde(default = "default_samples")]
    pub samples: usize,
    /// Fraction of image height used by text (rest is margin).
    #[serde(default = "default_fill")]
    pub fill_ratio: f64,
    /// RNG seed for reproducibility.
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// Path to word list file (one word per line). Empty = random chars.
    #[serde(default)]
    pub word_list: String,
    /// Case mode: "mixed" (default), "lower", "upper".
    #[serde(default = "default_case_mode")]
    pub case_mode: String,
}

fn default_height() -> usize { 128 }
fn default_width() -> usize { 256 }
fn default_min_letters() -> usize { 1 }
fn default_max_letters() -> usize { 4 }
fn default_samples() -> usize { 10000 }
fn default_fill() -> f64 { 0.80 }
fn default_seed() -> u64 { 0xF0D1_CAFE }
fn default_case_mode() -> String { "mixed".to_string() }

/// A loaded font with precomputed sizing.
struct LoadedFont {
    font: Font,
    px_size: f32,
    baseline_y: i32,
    name: String,
}

/// Generated dataset with raw pixel buffers for fast saving.
pub struct GeneratedDataset {
    pub dataset: WordDataset,
    pub pixel_buffers: Vec<Vec<f32>>,
    pub image_height: usize,
    pub image_width: usize,
}

/// Generate a `WordDataset` in memory from font files and a word list.
///
/// Each sample is a word image with proportionally-spaced glyphs.
/// GT centers are computed from actual font advance widths.
pub fn generate_word_dataset(cfg: &GenConfig) -> Result<GeneratedDataset> {
    let fonts = load_fonts(cfg)?;
    if fonts.is_empty() {
        return Err(TensorError::new("no fonts loaded"));
    }

    let words = load_word_list(cfg)?;
    let chars: Vec<char> = cfg.charset.chars().collect();

    eprintln!("Word generator: {} fonts, {} words, {} samples, case={}",
        fonts.len(), words.len(), cfg.samples, cfg.case_mode);

    let mut rng = cfg.seed;
    let mut samples = Vec::with_capacity(cfg.samples);
    let mut pixel_buffers = Vec::with_capacity(cfg.samples);
    let mut skipped = 0usize;

    let mut generated = 0;
    while generated < cfg.samples {
        let result = generate_one_word_sample(
            &fonts, &words, &chars, cfg, &mut rng,
        )?;
        if let Some((sample, pixels)) = result {
            samples.push(sample);
            pixel_buffers.push(pixels);
            generated += 1;
        } else {
            skipped += 1;
            if skipped > cfg.samples * 10 {
                return Err(TensorError::new("too many skipped samples — fonts may be too wide for canvas"));
            }
        }
    }

    if skipped > 0 {
        eprintln!("Skipped {} overflows ({:.1}%)", skipped,
            skipped as f64 / (generated + skipped) as f64 * 100.0);
    }
    eprintln!("Generated {} word samples in memory", samples.len());
    Ok(GeneratedDataset {
        dataset: WordDataset { samples },
        pixel_buffers,
        image_height: cfg.image_height,
        image_width: cfg.image_width,
    })
}

/// Letter-mode dataset: target at center, 0-2 random neighbors.
pub struct GeneratedLetterDataset {
    pub dataset: LetterDataset,
    pub pixel_buffers: Vec<Vec<f32>>,
    pub image_height: usize,
    pub image_width: usize,
}

/// Generate a `LetterDataset` with target letter centered and 0-2 neighbors.
///
/// Uses proportional spacing from font advance widths.
pub fn generate_letter_dataset(cfg: &GenConfig) -> Result<GeneratedLetterDataset> {
    let fonts = load_fonts(cfg)?;
    if fonts.is_empty() {
        return Err(TensorError::new("no fonts loaded"));
    }

    let chars: Vec<char> = cfg.charset.chars().collect();
    if chars.is_empty() {
        return Err(TensorError::new("empty charset"));
    }

    let h = cfg.image_height;
    let w = cfg.image_width;
    let mut rng = cfg.seed;

    // Balanced distribution: every (font, char) combo gets equal representation.
    let n_combos = fonts.len() * chars.len();
    let per_combo = cfg.samples / n_combos;
    let remainder = cfg.samples % n_combos;
    let total_base = per_combo * n_combos + remainder;

    eprintln!("Letter generator: {} fonts × {} chars = {} combos, {} per combo (+{} extra) = {} samples",
        fonts.len(), chars.len(), n_combos, per_combo, remainder, total_base);

    let mut raw_samples = Vec::with_capacity(total_base);
    let mut clean_by_id: HashMap<(i64, bool, String), Tensor> = HashMap::new();

    let mut combo_idx = 0usize;
    for lf in &fonts {
        for &target_ch in &chars {
            let letter_idx = char_to_idx(target_ch);
            let case_label = if target_ch.is_lowercase() { 1.0f32 } else { 0.0f32 };
            let repeats = per_combo + if combo_idx < remainder { 1 } else { 0 };
            combo_idx += 1;

            for _ in 0..repeats {
                let n = if cfg.min_letters == cfg.max_letters {
                    cfg.min_letters
                } else {
                    cfg.min_letters + rng_range(&mut rng, cfg.max_letters - cfg.min_letters + 1)
                };

                let mut pixels = vec![0.0f32; h * w];

                // Layout: target centered, neighbors spaced by advance width.
                let target_metrics = lf.font.metrics(target_ch, lf.px_size);
                let target_advance = target_metrics.advance_width;

                // Render target at center.
                let center_px = w as f32 / 2.0;
                let target_cursor = center_px - target_advance / 2.0;
                render_glyph_cursor(&lf.font, lf.px_size, lf.baseline_y, target_ch,
                                    target_cursor, &mut pixels, h, w);

                if n >= 2 {
                    let neighbor_ch = chars[rng_range(&mut rng, chars.len())];
                    let neighbor_advance = lf.font.metrics(neighbor_ch, lf.px_size).advance_width;
                    if n == 2 {
                        if rng_range(&mut rng, 2) == 0 {
                            // Left neighbor.
                            let cursor = target_cursor - neighbor_advance;
                            render_glyph_cursor(&lf.font, lf.px_size, lf.baseline_y, neighbor_ch,
                                                cursor, &mut pixels, h, w);
                        } else {
                            // Right neighbor.
                            let cursor = target_cursor + target_advance;
                            render_glyph_cursor(&lf.font, lf.px_size, lf.baseline_y, neighbor_ch,
                                                cursor, &mut pixels, h, w);
                        }
                    } else {
                        // Both neighbors.
                        let left_cursor = target_cursor - neighbor_advance;
                        render_glyph_cursor(&lf.font, lf.px_size, lf.baseline_y, neighbor_ch,
                                            left_cursor, &mut pixels, h, w);
                        let right_ch = chars[rng_range(&mut rng, chars.len())];
                        let right_cursor = target_cursor + target_advance;
                        render_glyph_cursor(&lf.font, lf.px_size, lf.baseline_y, right_ch,
                                            right_cursor, &mut pixels, h, w);
                    }
                }

                let clean = Tensor::from_f32(&pixels, &[1, h as i64, w as i64], Device::CPU)?;
                let is_lower = case_label > 0.5;
                clean_by_id.entry((letter_idx, is_lower, lf.name.clone()))
                    .or_insert_with(|| clean.clone());

                raw_samples.push((clean, letter_idx, case_label, lf.name.clone()));
            }
        }
    }

    // Resolve partners (opposite case, same letter, same font).
    let mut has_partners = true;
    let mut samples = Vec::with_capacity(raw_samples.len());
    let mut pixel_buffers = Vec::with_capacity(raw_samples.len());

    for (clean, letter_idx, case_label, font_name) in raw_samples {
        let is_lower = case_label > 0.5;
        let partner_clean = if let Some(p) = clean_by_id.get(&(letter_idx, !is_lower, font_name.clone())) {
            p.clone()
        } else {
            has_partners = false;
            Tensor::zeros(&clean.shape(), Default::default())?
        };

        let pixels = clean.to_f32_vec()?;
        pixel_buffers.push(pixels);

        samples.push(LetterSample {
            image: clean.clone(),
            clean,
            letter_idx,
            case_label,
            font: font_name,
            partner_clean,
        });
    }

    if !has_partners {
        eprintln!("Note: some letters missing opposite-case partners");
    }
    eprintln!("Generated {} letter samples in memory ({} with partners)",
        samples.len(), if has_partners { "all" } else { "some" });

    Ok(GeneratedLetterDataset {
        dataset: LetterDataset { samples, has_partners },
        pixel_buffers,
        image_height: h,
        image_width: w,
    })
}

/// Save a generated word dataset to disk (PNGs + metadata.json).
pub fn save_dataset(gds: &GeneratedDataset, dir: &str) -> Result<()> {
    let dir_path = Path::new(dir);
    fs::create_dir_all(dir_path)
        .map_err(|e| TensorError::new(&format!("create dir: {e}")))?;

    let mut meta: HashMap<String, serde_json::Value> = HashMap::new();

    for (i, sample) in gds.dataset.samples.iter().enumerate() {
        let img_name = format!("img_{:05}.png", i);
        let img_path = dir_path.join(&img_name);

        save_tensor_png(&gds.pixel_buffers[i], gds.image_height, gds.image_width, &img_path)?;

        let letters: Vec<String> = sample.letters.iter().map(|&idx| {
            if idx >= 0 { ((idx as u8 + b'A') as char).to_string() }
            else { "_".to_string() }
        }).collect();

        meta.insert(img_name.clone(), serde_json::json!({
            "image": img_name,
            "clean": img_name,
            "word": sample.word,
            "letters": letters,
            "centers": sample.centers,
        }));
    }

    let meta_path = dir_path.join("metadata.json");
    fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap_or_default())
        .map_err(|e| TensorError::new(&format!("write metadata: {e}")))?;

    eprintln!("Saved {} samples to {}", gds.dataset.samples.len(), dir);
    Ok(())
}

// ── Word generation ─────────────────────────────────────────────────

/// Generate one word sample with proportional glyph spacing.
/// Returns None if the word overflows the canvas.
fn generate_one_word_sample(
    fonts: &[LoadedFont],
    words: &[String],
    chars: &[char],
    cfg: &GenConfig,
    rng: &mut u64,
) -> Result<Option<(WordSample, Vec<f32>)>> {
    let h = cfg.image_height;
    let w = cfg.image_width;

    let font_idx = rng_range(rng, fonts.len());
    let lf = &fonts[font_idx];

    // Pick a word and apply case.
    let base_word = if !words.is_empty() {
        words[rng_range(rng, words.len())].clone()
    } else {
        // Random char fallback.
        let n = if cfg.min_letters == cfg.max_letters {
            cfg.min_letters
        } else {
            cfg.min_letters + rng_range(rng, cfg.max_letters - cfg.min_letters + 1)
        };
        (0..n).map(|_| chars[rng_range(rng, chars.len())]).collect()
    };

    let word = apply_case(&base_word, &cfg.case_mode, rng);
    let word_chars: Vec<char> = word.chars().collect();

    if word_chars.is_empty() {
        return Ok(None);
    }

    // Layout pass: rasterize all glyphs, compute positions.
    let mut glyphs: Vec<(fontdue::Metrics, Vec<u8>)> = Vec::with_capacity(word_chars.len());
    let mut total_advance = 0.0f32;

    for &ch in &word_chars {
        let (metrics, bitmap) = lf.font.rasterize(ch, lf.px_size);
        total_advance += metrics.advance_width;
        glyphs.push((metrics, bitmap));
    }

    // Skip if word overflows canvas (with some margin).
    if total_advance > w as f32 * 0.95 {
        return Ok(None);
    }

    // Center the word on the canvas.
    let start_x = (w as f32 - total_advance) / 2.0;
    let mut cursor = start_x;

    let mut pixels = vec![0.0f32; h * w];
    let mut letters = Vec::with_capacity(word_chars.len());
    let mut centers = Vec::with_capacity(word_chars.len());

    for (i, &ch) in word_chars.iter().enumerate() {
        let (ref metrics, ref bitmap) = glyphs[i];
        letters.push(char_to_idx(ch));

        // Center of this glyph in pixel space.
        let center_px = cursor + metrics.advance_width / 2.0;
        // Normalize: pixel / width * 2 - 1.
        let norm_x = (center_px as f64 / w as f64) * 2.0 - 1.0;
        centers.push(norm_x);

        // Render from pre-rasterized bitmap.
        render_from_bitmap(
            metrics, bitmap, lf.baseline_y, cursor,
            &mut pixels, h, w,
        );

        cursor += metrics.advance_width;
    }

    let image = Tensor::from_f32(&pixels, &[1, h as i64, w as i64], Device::CPU)?;

    Ok(Some((WordSample {
        image: image.clone(),
        clean: image,
        word,
        letters,
        centers,
    }, pixels)))
}

/// Apply random casing to a word.
fn apply_case(word: &str, mode: &str, rng: &mut u64) -> String {
    match mode {
        "lower" => word.to_lowercase(),
        "upper" => word.to_uppercase(),
        _ => {
            // Mixed: ~50% lower, ~25% upper, ~25% title.
            match rng_range(rng, 4) {
                0 => word.to_uppercase(),
                1 => {
                    let mut chars = word.chars();
                    match chars.next() {
                        None => String::new(),
                        Some(c) => c.to_uppercase().to_string() + &chars.as_str().to_lowercase(),
                    }
                }
                _ => word.to_lowercase(),
            }
        }
    }
}

// ── Word list loading ───────────────────────────────────────────────

fn load_word_list(cfg: &GenConfig) -> Result<Vec<String>> {
    if cfg.word_list.is_empty() {
        return Ok(Vec::new()); // random char fallback
    }

    let text = fs::read_to_string(&cfg.word_list)
        .map_err(|e| TensorError::new(&format!("read word list '{}': {e}", cfg.word_list)))?;

    let words: Vec<String> = text.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter(|l| {
            let n = l.chars().count();
            n >= cfg.min_letters && n <= cfg.max_letters
        })
        .map(|l| l.to_string())
        .collect();

    if words.is_empty() {
        return Err(TensorError::new(&format!(
            "no words with {}-{} letters in '{}'", cfg.min_letters, cfg.max_letters, cfg.word_list
        )));
    }

    eprintln!("Loaded {} words from '{}' ({}-{} letters)",
        words.len(), cfg.word_list, cfg.min_letters, cfg.max_letters);
    Ok(words)
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

/// Render a glyph at a pixel cursor position (left edge of advance box).
fn render_glyph_cursor(
    font: &Font, px_size: f32, baseline_y: i32, ch: char,
    cursor_x: f32, pixels: &mut [f32], img_h: usize, img_w: usize,
) {
    let (metrics, bitmap) = font.rasterize(ch, px_size);
    render_from_bitmap(&metrics, &bitmap, baseline_y, cursor_x, pixels, img_h, img_w);
}

/// Render a pre-rasterized glyph bitmap at a cursor position.
fn render_from_bitmap(
    metrics: &fontdue::Metrics, bitmap: &[u8],
    baseline_y: i32, cursor_x: f32,
    pixels: &mut [f32], img_h: usize, img_w: usize,
) {
    if metrics.width == 0 || metrics.height == 0 { return; }

    let glyph_x = cursor_x as i32 + metrics.xmin;
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

/// Save a float pixel buffer as a grayscale PNG.
pub fn save_letter_png(pixels: &[f32], h: usize, w: usize, path: &Path) -> Result<()> {
    save_tensor_png(pixels, h, w, path)
}

fn save_tensor_png(pixels: &[f32], h: usize, w: usize, path: &Path) -> Result<()> {
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

// ── RNG helpers ──────────────────────────────────────────────────────

#[inline]
fn rng_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

#[inline]
fn rng_range(state: &mut u64, n: usize) -> usize {
    (rng_next(state) >> 33) as usize % n
}
