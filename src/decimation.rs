//! decimation.

// Decimation & interpolation
// ---------------------------------------------------------------------------

/// Decimate a signal by factor `m` (take every `m`-th sample).
///
/// # Panics
///
/// Panics if `factor` is zero.
#[must_use]
pub fn decimate(signal: &[f64], factor: usize) -> Vec<f64> {
    assert!(factor > 0, "decimation factor must be > 0");
    signal.iter().step_by(factor).copied().collect()
}

/// Interpolate a signal by factor `m` (zero-insertion).
///
/// # Panics
///
/// Panics if `factor` is zero.
#[must_use]
pub fn interpolate(signal: &[f64], factor: usize) -> Vec<f64> {
    assert!(factor > 0, "interpolation factor must be > 0");
    let mut out = vec![0.0; signal.len() * factor];
    for (i, &s) in signal.iter().enumerate() {
        out[i * factor] = s;
    }
    out
}

/// Linear interpolation upsample by factor `m`.
///
/// # Panics
///
/// Panics if `factor` is zero.
#[must_use]
pub fn interpolate_linear(signal: &[f64], factor: usize) -> Vec<f64> {
    assert!(factor > 0, "interpolation factor must be > 0");
    if signal.len() < 2 {
        return signal.to_vec();
    }
    let out_len = (signal.len() - 1) * factor + 1;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..signal.len() - 1 {
        let a = signal[i];
        let b = signal[i + 1];
        for k in 0..factor {
            let t = k as f64 / factor as f64;
            out.push(a.mul_add(1.0 - t, b * t));
        }
    }
    out.push(*signal.last().unwrap_or(&0.0));
    out
}
