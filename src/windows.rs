//! windows.

use std::f64::consts::PI;

// Window functions
// ---------------------------------------------------------------------------

/// Generate a Hamming window of length `n`.
#[must_use]
pub fn hamming(n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0; n];
    }
    let m = (n - 1) as f64;
    (0..n)
        .map(|i| 0.46f64.mul_add(-(2.0 * PI * i as f64 / m).cos(), 0.54))
        .collect()
}

/// Generate a Hanning (Hann) window of length `n`.
#[must_use]
pub fn hanning(n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0; n];
    }
    let m = (n - 1) as f64;
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / m).cos()))
        .collect()
}

/// Generate a Blackman window of length `n`.
#[must_use]
pub fn blackman(n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0; n];
    }
    let m = (n - 1) as f64;
    (0..n)
        .map(|i| {
            let x = i as f64;
            0.08f64.mul_add(
                (4.0 * PI * x / m).cos(),
                0.5f64.mul_add(-(2.0 * PI * x / m).cos(), 0.42),
            )
        })
        .collect()
}
