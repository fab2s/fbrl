//! Word recognition domain — modules, training phases, data, and losses.
//!
//! ## Training phases
//!
//! **Phase 2: SubScan + Letter**
//! ```text
//! word image → SubScan(bounded region) → position
//!            → LetterModel.scan(from position) → LetterModel.read(frozen) → classify
//! ```
//!
//! **Phase 3: Full word model (future)**
//! ```text
//! word image → MetaScan → N regions
//!            → each(N): SubScan → LetterModel → per-position classification
//! ```

pub mod data;
pub mod glimpse;
pub mod loss;
pub mod subscan;
pub mod subscan_train;
pub mod synthetic;
// pub mod subscan_eval;   // step 2 evaluation (TODO)
// pub mod model;          // full word model (step 3, future)
// pub mod train;          // step 3 training loop (future)
// pub mod eval;           // step 3 evaluation (future)

pub use data::*;
pub use glimpse::*;
pub use loss::*;
pub use subscan::*;
pub use subscan_train::*;
pub use synthetic::*;
