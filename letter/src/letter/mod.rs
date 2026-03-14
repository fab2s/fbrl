//! Letter recognition domain — model, training, data, and losses.
//!
//! The pipeline (v2, with scan phase):
//!
//! ```text
//! image → H0Init → Loop(ScanStep).Using("image").Tag("scan")
//!       → Loop(AttentionStep).Using("image").Tag("read")
//!       → Linear → Fork(letterHead) → Fork(caseHead)
//!       → Decoder.Using("latent", "case")
//! ```
//!
//! Scan and read fixations are collected separately via graph traces.

mod data;
mod decoder;
mod glimpse;
mod loss;
mod model;
mod modules;
mod synthetic;
mod train;

pub use data::*;
pub use decoder::*;
pub use glimpse::*;
pub use loss::*;
pub use model::*;
pub use modules::*;
pub use synthetic::*;
pub use train::*;
