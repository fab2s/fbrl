//! Synthetic word data for smoke tests.

use flodl::tensor::{Result, Tensor};

use super::data::WordSample;

/// Generate N random word samples with variable lengths (1-4 letters).
pub fn new_synthetic_dataset(n: usize) -> Result<Vec<WordSample>> {
    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        let image = Tensor::rand(&[1, 128, 256], Default::default())?;
        let clean = Tensor::rand(&[1, 128, 256], Default::default())?;
        let n_letters = 1 + (rand_u8() % 4) as usize;
        let letters: Vec<i64> = (0..n_letters).map(|_| (rand_u8() % 26) as i64).collect();
        let centers: Vec<f64> = (0..n_letters).map(|i| {
            (i as f64 + 0.5) / n_letters as f64 * 2.0 - 1.0
        }).collect();
        let word: String = letters.iter().map(|&l| (b'a' + l as u8) as char).collect();
        samples.push(WordSample {
            image, clean, letters, centers, word,
        });
    }
    Ok(samples)
}

fn rand_u8() -> u8 {
    static mut STATE: u32 = 42;
    unsafe {
        STATE = STATE.wrapping_mul(1103515245).wrapping_add(12345);
        (STATE >> 16) as u8
    }
}
