//! utility.

// Utility
// ---------------------------------------------------------------------------

/// Compute the energy of a signal.
#[must_use]
pub fn energy(signal: &[f64]) -> f64 {
    signal.iter().map(|x| x * x).sum()
}

/// Compute the RMS (root mean square) of a signal.
#[must_use]
pub fn rms(signal: &[f64]) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }
    (energy(signal) / signal.len() as f64).sqrt()
}

/// Zero-pad a signal to the next power of two.
#[must_use]
pub fn zero_pad_to_power_of_two(signal: &[f64]) -> Vec<f64> {
    if signal.is_empty() {
        return vec![0.0];
    }
    let n = signal.len().next_power_of_two();
    let mut out = vec![0.0; n];
    out[..signal.len()].copy_from_slice(signal);
    out
}
