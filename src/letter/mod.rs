//! Letter recognition domain — model, training, data, and losses.
//!
//! The pipeline:
//!
//! ```text
//! image → H0Init → Loop(AttentionStep).Using("image")
//!       → LatentHead → Split(letterHead, caseHead)
//!       → Merge → Decoder.Using("latent", "case")
//! ```
//!
//! Fixation locations are collected automatically via graph traces.

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
