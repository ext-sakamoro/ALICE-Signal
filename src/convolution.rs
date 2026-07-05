//! convolution.

// Convolution & correlation
// ---------------------------------------------------------------------------

/// Linear convolution of two signals.
#[must_use]
pub fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let out_len = a.len() + b.len() - 1;
    let mut result = vec![0.0; out_len];
    for (i, &av) in a.iter().enumerate() {
        for (j, &bv) in b.iter().enumerate() {
            result[i + j] = av.mul_add(bv, result[i + j]);
        }
    }
    result
}

/// Cross-correlation of two signals.
#[must_use]
pub fn correlate(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let out_len = a.len() + b.len() - 1;
    let mut result = vec![0.0; out_len];
    let offset = b.len() - 1;
    for (i, &av) in a.iter().enumerate() {
        for (j, &bv) in b.iter().enumerate() {
            let idx = i + offset - j;
            result[idx] = av.mul_add(bv, result[idx]);
        }
    }
    result
}
