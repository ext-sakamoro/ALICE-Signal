//! psd.

use crate::complex::*;
use crate::fft::*;

// Power spectral density
// ---------------------------------------------------------------------------

/// Estimate the power spectral density using the periodogram method.
///
/// Returns `N/2 + 1` values (one-sided PSD for real input).
///
/// # Panics
///
/// Panics if `signal` length is not a power of two.
#[must_use]
pub fn psd(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    assert!(is_power_of_two(n), "PSD input length must be power of two");

    let mut buf: Vec<Complex> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft(&mut buf);

    let half = n / 2 + 1;
    let scale = 1.0 / n as f64;
    (0..half).map(|i| buf[i].mag_sq() * scale).collect()
}

/// Estimate PSD with a window applied before FFT.
///
/// # Panics
///
/// Panics if `signal` length is not a power of two or window length mismatches.
#[must_use]
pub fn psd_windowed(signal: &[f64], window: &[f64]) -> Vec<f64> {
    assert_eq!(
        signal.len(),
        window.len(),
        "window length must match signal"
    );
    let windowed: Vec<f64> = signal.iter().zip(window).map(|(s, w)| s * w).collect();
    psd(&windowed)
}
