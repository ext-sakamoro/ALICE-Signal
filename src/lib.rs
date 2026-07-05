//! ALICE-Signal: Pure Rust digital signal processing library.

#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::module_name_repetitions,
    clippy::needless_range_loop,
    clippy::useless_vec,
    clippy::manual_range_contains,
    clippy::missing_const_for_fn,
    clippy::match_same_arms,
    clippy::should_panic_without_expect,
    clippy::suboptimal_flops,
    clippy::redundant_clone,
    clippy::wildcard_imports,
    clippy::too_many_lines,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::items_after_statements,
    clippy::excessive_precision,
    clippy::unreadable_literal,
    clippy::float_cmp,
    clippy::manual_midpoint,
    clippy::approx_constant
)]

pub mod batch_verify;
pub mod complex;
pub mod convolution;
pub mod correlator;
pub mod decimation;
pub mod ekf;
pub mod fft;
pub mod fir;
pub mod gnss;
pub mod iir;
pub mod kalman;
pub mod prelude;
pub mod psd;
pub mod spoofing;
pub mod tracking;
pub mod utility;
pub mod wavelet;
pub mod windows;

#[cfg(test)]
mod integration_tests;

pub use crate::complex::*;
pub use crate::convolution::*;
pub use crate::decimation::*;
pub use crate::fft::*;
pub use crate::fir::*;
pub use crate::iir::*;
pub use crate::psd::*;
pub use crate::utility::*;
pub use crate::wavelet::*;
pub use crate::windows::*;
