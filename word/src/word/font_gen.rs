//! In-memory font-based dataset generator.
//!
//! Loads TTF/OTF fonts, rasterizes letters, and composes word images
//! with variable neighbor counts. Returns a `WordDataset` directly
//! in RAM — no disk I/O needed. Optional `--save` dumps PNGs + metadata.json.
//!
//! Images are white-on-black (ink=1.0, background=0.0) to match the
//! existing training data convention.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use fontdue::{Font, FontSettings};
use flodl::tensor::{Device, Result, Tensor, TensorError};
use serde::{Serialize, Deserialize};

use fbrl::letter::{LetterDataset, LetterSample};
use super::data::{WordDataset, WordSample};

/// Number of letter positions in a word image.
const N_POSITIONS: usize = 4;

/// Normalized x-centers for the 4 letter positions in a 128x256 word image.
const LETTER_CENTERS: [f64; N_POSITIONS] = [-0.75, -0.25, 0.25, 0.75];

/// Generator configuration — loaded from JSON.
#[derive(Serialize, Deserialize, Clone)]
pub struct GenConfig {
    /// Glob pattern for TTF/OTF font files.
    pub fonts: String,
    /// Characters to use (e.g., "abcdefghijklmnopqrstuvwxyz").
    pub charset: String,
    /// Image height in pixels.
    #[serde(default = "default_height")]
    pub image_height: usize,
    /// Image width in pixels.
    #[serde(default = "default_width")]
    pub image_width: usize,
    /// Minimum letters per image (1-4).
    #[serde(default = "default_min_letters")]
    pub min_letters: usize,
    /// Maximum letters per image (1-4).
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
}

fn default_height() -> usize { 128 }
fn default_width() -> usize { 256 }
fn default_min_letters() -> usize { 1 }
fn default_max_letters() -> usize { 4 }
fn default_samples() -> usize { 10000 }
fn default_fill() -> f64 { 0.80 }
fn default_seed() -> u64 { 0xF0D1_CAFE }

/// A loaded font with precomputed sizing.
struct LoadedFont {
    font: Font,
    px_size: f32,
    baseline_y: i32, // pixel y of baseline from top of image
    name: String,
}

/// Generated dataset with raw pixel buffers for fast saving.
pub struct GeneratedDataset {
    pub dataset: WordDataset,
    /// Raw pixel buffers (one per sample) for PNG saving without tensor extraction.
    pub pixel_buffers: Vec<Vec<f32>>,
    pub image_height: usize,
    pub image_width: usize,
}

/// Generate a `WordDataset` in memory from font files.
///
/// Each sample is a 128×256 image with 1-4 letters placed at standard
/// word positions. Empty positions have letter_idx = -1.
pub fn generate_word_dataset(cfg: &GenConfig) -> Result<GeneratedDataset> {
    let fonts = load_fonts(cfg)?;
    if fonts.is_empty() {
        return Err(TensorError::new("no fonts loaded"));
    }

    let chars: Vec<char> = cfg.charset.chars().collect();
    if chars.is_empty() {
        return Err(TensorError::new("empty charset"));
    }

    eprintln!("Generator: {} fonts, {} chars, {} samples, {}-{} letters/image",
        fonts.len(), chars.len(), cfg.samples, cfg.min_letters, cfg.max_letters);

    let mut rng = cfg.seed;
    let mut samples = Vec::with_capacity(cfg.samples);
    let mut pixel_buffers = Vec::with_capacity(cfg.samples);

    for _ in 0..cfg.samples {
        let (sample, pixels) = generate_one_sample(
            &fonts, &chars, cfg, &mut rng,
        )?;
        samples.push(sample);
        pixel_buffers.push(pixels);
    }

    eprintln!("Generated {} samples in memory", samples.len());
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
/// Uses the same GenConfig. `min_letters`/`max_letters` control neighbor count
/// (1 = alone, 2 = one neighbor, 3 = both neighbors). Images are 128×256
/// matching word-image format for inference compatibility.
pub fn generate_letter_dataset(cfg: &GenConfig) -> Result<GeneratedLetterDataset> {
    let fonts = load_fonts(cfg)?;
    if fonts.is_empty() {
        return Err(TensorError::new("no fonts loaded"));
    }

    let chars: Vec<char> = cfg.charset.chars().collect();
    if chars.is_empty() {
        return Err(TensorError::new("empty charset"));
    }

    // Letter spacing in normalized coords (matches word layout).
    let spacing = 0.5;
    // Positions: center=0.0, left=-spacing, right=+spacing.
    let center_norm = 0.0;
    let left_norm = -spacing;
    let right_norm = spacing;

    eprintln!("Letter generator: {} fonts, {} chars, {} samples, {}-{} letters/image",
        fonts.len(), chars.len(), cfg.samples, cfg.min_letters, cfg.max_letters);

    let mut rng = cfg.seed;
    let mut samples = Vec::with_capacity(cfg.samples);
    let mut pixel_buffers = Vec::with_capacity(cfg.samples);
    let h = cfg.image_height;
    let w = cfg.image_width;

    for _ in 0..cfg.samples {
        let font_idx = rng_range(&mut rng, fonts.len());
        let lf = &fonts[font_idx];

        // Pick target letter.
        let ch_idx = rng_range(&mut rng, chars.len());
        let target_ch = chars[ch_idx];
        let letter_idx = char_to_idx(target_ch);
        let case_label = if target_ch.is_lowercase() { 1.0f32 } else { 0.0f32 };

        // Pick number of total letters (1 = alone, 2 = one neighbor, 3 = both).
        let n = if cfg.min_letters == cfg.max_letters {
            cfg.min_letters
        } else {
            cfg.min_letters + rng_range(&mut rng, cfg.max_letters - cfg.min_letters + 1)
        };

        let mut pixels = vec![0.0f32; h * w];

        // Always render target at center.
        render_glyph_at(&lf.font, lf.px_size, lf.baseline_y, target_ch,
                        center_norm, &mut pixels, h, w);

        if n >= 2 {
            // At least one neighbor. Pick left or right (or both if n=3).
            let neighbor_ch = chars[rng_range(&mut rng, chars.len())];
            if n == 2 {
                // One neighbor: randomly left or right.
                let pos = if rng_range(&mut rng, 2) == 0 { left_norm } else { right_norm };
                render_glyph_at(&lf.font, lf.px_size, lf.baseline_y, neighbor_ch,
                                pos, &mut pixels, h, w);
            } else {
                // Both neighbors.
                render_glyph_at(&lf.font, lf.px_size, lf.baseline_y, neighbor_ch,
                                left_norm, &mut pixels, h, w);
                let right_ch = chars[rng_range(&mut rng, chars.len())];
                render_glyph_at(&lf.font, lf.px_size, lf.baseline_y, right_ch,
                                right_norm, &mut pixels, h, w);
            }
        }

        let image = Tensor::from_f32(&pixels, &[1, h as i64, w as i64], Device::CPU)?;

        samples.push(LetterSample {
            image: image.clone(),
            clean: image.clone(),
            letter_idx,
            case_label,
            partner_clean: image, // dummy — train with recode_weight=0
        });
        pixel_buffers.push(pixels);
    }

    eprintln!("Generated {} letter samples in memory", samples.len());
    Ok(GeneratedLetterDataset {
        dataset: LetterDataset { samples, has_partners: false },
        pixel_buffers,
        image_height: h,
        image_width: w,
    })
}

/// Save a generated dataset to disk (PNGs + metadata.json).
pub fn save_dataset(gds: &GeneratedDataset, dir: &str) -> Result<()> {
    let dir_path = Path::new(dir);
    fs::create_dir_all(dir_path)
        .map_err(|e| TensorError::new(&format!("create dir: {e}")))?;

    let mut meta: HashMap<String, serde_json::Value> = HashMap::new();

    for (i, sample) in gds.dataset.samples.iter().enumerate() {
        let img_name = format!("img_{:05}.png", i);
        let img_path = dir_path.join(&img_name);

        // Write image from raw pixel buffer (fast — no tensor extraction).
        save_tensor_png(&gds.pixel_buffers[i], gds.image_height, gds.image_width, &img_path)?;

        // Metadata entry (clean = image for generated data).
        let letters: Vec<String> = sample.letters.iter().map(|&idx| {
            if idx >= 0 { ((idx as u8 + b'a') as char).to_string() }
            else { "_".to_string() }
        }).collect();

        meta.insert(img_name.clone(), serde_json::json!({
            "image": img_name,
            "clean": img_name,
            "word": sample.word,
            "letter1": letters[0],
            "letter2": letters[1],
            "letter3": letters[2],
            "letter4": letters[3],
        }));
    }

    let meta_path = dir_path.join("metadata.json");
    fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap_or_default())
        .map_err(|e| TensorError::new(&format!("write metadata: {e}")))?;

    eprintln!("Saved {} samples to {}", gds.dataset.samples.len(), dir);
    Ok(())
}

// ── Internal ─────────────────────────────────────────────────────────

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

        // Compute px_size so line height fills `fill_ratio` of image.
        let target_px = (cfg.image_height as f64 * cfg.fill_ratio) as f32;

        // Start with a guess and refine.
        let mut px_size = target_px;
        for _ in 0..5 {
            let lm = font.horizontal_line_metrics(px_size);
            if let Some(lm) = lm {
                let actual_height = lm.ascent - lm.descent;
                if actual_height > 0.1 {
                    px_size = px_size * target_px / actual_height;
                }
            }
        }

        // Compute baseline position (pixels from top of image).
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

fn generate_one_sample(
    fonts: &[LoadedFont],
    chars: &[char],
    cfg: &GenConfig,
    rng: &mut u64,
) -> Result<(WordSample, Vec<f32>)> {
    let h = cfg.image_height;
    let w = cfg.image_width;

    // Pick random font.
    let font_idx = rng_range(rng, fonts.len());
    let lf = &fonts[font_idx];

    // Pick number of letters.
    let n_letters = if cfg.min_letters == cfg.max_letters {
        cfg.min_letters
    } else {
        cfg.min_letters + rng_range(rng, cfg.max_letters - cfg.min_letters + 1)
    };

    // Pick which positions to fill (contiguous block).
    let max_start = N_POSITIONS.saturating_sub(n_letters);
    let start_pos = if max_start > 0 { rng_range(rng, max_start + 1) } else { 0 };

    // Pick letters for each position.
    let mut letters = [-1i64; N_POSITIONS];
    let mut word = String::new();
    for i in 0..n_letters {
        let pos = start_pos + i;
        let ch_idx = rng_range(rng, chars.len());
        let ch = chars[ch_idx];
        letters[pos] = char_to_idx(ch);
        word.push(ch);
    }

    // Render to pixel buffer.
    let mut pixels = vec![0.0f32; h * w];

    for pos in 0..N_POSITIONS {
        if letters[pos] < 0 { continue; }
        let ch = idx_to_char(letters[pos], chars);
        render_glyph(&lf.font, lf.px_size, lf.baseline_y, ch, pos,
                      &mut pixels, h, w);
    }

    // Create tensors.
    let image = Tensor::from_f32(&pixels, &[1, h as i64, w as i64], Device::CPU)?;

    Ok((WordSample {
        image: image.clone(),
        clean: image,
        letters,
        word,
    }, pixels))
}

fn render_glyph(
    font: &Font, px_size: f32, baseline_y: i32, ch: char,
    position: usize, pixels: &mut [f32], img_h: usize, img_w: usize,
) {
    render_glyph_at(font, px_size, baseline_y, ch,
                    LETTER_CENTERS[position], pixels, img_h, img_w);
}

fn render_glyph_at(
    font: &Font, px_size: f32, baseline_y: i32, ch: char,
    norm_x: f64, pixels: &mut [f32], img_h: usize, img_w: usize,
) {
    let (metrics, bitmap) = font.rasterize(ch, px_size);
    if metrics.width == 0 || metrics.height == 0 { return; }

    // Normalized x → pixel center.
    let center_x = ((norm_x + 1.0) / 2.0 * img_w as f64) as i32;

    // Glyph pixel origin.
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

fn idx_to_char(idx: i64, chars: &[char]) -> char {
    // Find a char in charset matching this letter index.
    let target_upper = (idx as u8 + b'A') as char;
    for &ch in chars {
        if ch.to_ascii_uppercase() == target_upper {
            return ch;
        }
    }
    target_upper
}

/// Save a float pixel buffer as a grayscale PNG.
pub fn save_letter_png(pixels: &[f32], h: usize, w: usize, path: &Path) -> Result<()> {
    save_tensor_png(pixels, h, w, path)
}

fn save_tensor_png(pixels: &[f32], h: usize, w: usize, path: &Path) -> Result<()> {
    // Convert float pixels [0, 1] to u8 [0, 255].
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
