//! fir.

use crate::convolution::convolve;
use crate::windows::*;
use std::f64::consts::PI;

// FIR filter design (windowed sinc)
// ---------------------------------------------------------------------------

/// Design a low-pass FIR filter using windowed sinc method with a Hamming window.
///
/// - `order`: filter order (number of taps will be `order + 1`)
/// - `cutoff`: normalised cutoff frequency (0.0 .. 1.0, where 1.0 = Nyquist)
///
/// # Panics
///
/// Panics if `cutoff` is not in `(0.0, 1.0)`.
#[must_use]
pub fn fir_lowpass(order: usize, cutoff: f64) -> Vec<f64> {
    assert!(cutoff > 0.0 && cutoff < 1.0, "cutoff must be in (0.0, 1.0)");

    let taps = order + 1;
    let mid = order as f64 / 2.0;
    let wc = PI * cutoff;
    let win = hamming(taps);

    let mut coeffs: Vec<f64> = (0..taps)
        .map(|i| {
            let n = i as f64 - mid;
            let sinc = if n.abs() < 1e-12 {
                wc / PI
            } else {
                (wc * n).sin() / (PI * n)
            };
            sinc * win[i]
        })
        .collect();

    // Normalise DC gain to 1
    let sum: f64 = coeffs.iter().sum();
    if sum.abs() > 1e-15 {
        for c in &mut coeffs {
            *c /= sum;
        }
    }
    coeffs
}

/// Design a high-pass FIR filter via spectral inversion of a low-pass filter.
#[must_use]
pub fn fir_highpass(order: usize, cutoff: f64) -> Vec<f64> {
    let mut h = fir_lowpass(order, cutoff);
    for c in &mut h {
        *c = -*c;
    }
    let mid = order / 2;
    h[mid] += 1.0;
    h
}

/// Apply an FIR filter to the input signal (direct convolution).
#[must_use]
pub fn fir_filter(signal: &[f64], coeffs: &[f64]) -> Vec<f64> {
    convolve(signal, coeffs)
}
