//! Synthetic word data for smoke tests.

use flodl::tensor::{Result, Tensor};

use super::data::WordSample;

/// Generate N random word samples.
pub fn new_synthetic_dataset(n: usize) -> Result<Vec<WordSample>> {
    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        let image = Tensor::rand(&[1, 128, 256], Default::default())?;
        let clean = Tensor::rand(&[1, 128, 256], Default::default())?;
        let letters = [
            (rand_u8() % 26) as i64,
            (rand_u8() % 26) as i64,
            (rand_u8() % 26) as i64,
            (rand_u8() % 26) as i64,
        ];
        samples.push(WordSample {
            image, clean, letters,
            word: format!("{}{}{}{}",
                (b'a' + letters[0] as u8) as char,
                (b'a' + letters[1] as u8) as char,
                (b'a' + letters[2] as u8) as char,
                (b'a' + letters[3] as u8) as char,
            ),
        });
    }
    Ok(samples)
}

fn rand_u8() -> u8 {
    // Simple deterministic-enough source for tests.
    static mut STATE: u32 = 42;
    unsafe {
        STATE = STATE.wrapping_mul(1103515245).wrapping_add(12345);
        (STATE >> 16) as u8
    }
}
