//! wavelet.

use crate::fft::is_power_of_two;

// Wavelet transforms
// ---------------------------------------------------------------------------

/// Haar wavelet forward transform (in-place, one level).
///
/// # Panics
///
/// Panics if `data.len()` is not a power of two or is less than 2.
pub fn haar_forward(data: &mut [f64]) {
    let n = data.len();
    assert!(
        is_power_of_two(n) && n >= 2,
        "length must be power of 2 and >= 2"
    );

    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    let half = n / 2;
    let mut temp = vec![0.0; n];
    for i in 0..half {
        temp[i] = (data[2 * i] + data[2 * i + 1]) * inv_sqrt2;
        temp[half + i] = (data[2 * i] - data[2 * i + 1]) * inv_sqrt2;
    }
    data.copy_from_slice(&temp);
}

/// Haar wavelet inverse transform (in-place, one level).
///
/// # Panics
///
/// Panics if `data.len()` is not a power of two or is less than 2.
pub fn haar_inverse(data: &mut [f64]) {
    let n = data.len();
    assert!(
        is_power_of_two(n) && n >= 2,
        "length must be power of 2 and >= 2"
    );

    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    let half = n / 2;
    let mut temp = vec![0.0; n];
    for i in 0..half {
        temp[2 * i] = (data[i] + data[half + i]) * inv_sqrt2;
        temp[2 * i + 1] = (data[i] - data[half + i]) * inv_sqrt2;
    }
    data.copy_from_slice(&temp);
}

/// Multi-level Haar wavelet forward transform.
///
/// # Panics
///
/// Panics if `data.len()` is not a power of two.
pub fn haar_forward_multi(data: &mut [f64]) {
    let n = data.len();
    assert!(is_power_of_two(n), "length must be power of 2");
    let mut len = n;
    while len >= 2 {
        haar_forward(&mut data[..len]);
        len /= 2;
    }
}

/// Multi-level Haar wavelet inverse transform.
///
/// # Panics
///
/// Panics if `data.len()` is not a power of two.
pub fn haar_inverse_multi(data: &mut [f64]) {
    let n = data.len();
    assert!(is_power_of_two(n), "length must be power of 2");
    let mut len = 2;
    while len <= n {
        haar_inverse(&mut data[..len]);
        len *= 2;
    }
}

/// Daubechies-4 wavelet coefficients.
const DB4_H: [f64; 4] = [
    0.482_962_913_144_534_16,
    0.836_516_303_737_807_9,
    0.224_143_868_042_013_4,
    -0.129_409_522_551_260_37,
];

/// Daubechies-4 forward wavelet transform (one level).
///
/// # Panics
///
/// Panics if `data.len()` is less than 4 or not even.
pub fn db4_forward(data: &mut [f64]) {
    let n = data.len();
    assert!(
        n >= 4 && n.is_multiple_of(2),
        "length must be >= 4 and even"
    );

    let half = n / 2;
    let mut approx = vec![0.0; half];
    let mut detail = vec![0.0; half];

    for i in 0..half {
        for (k, &hk) in DB4_H.iter().enumerate() {
            let idx = (2 * i + k) % n;
            approx[i] += hk * data[idx];
        }
        // High-pass: alternate sign reversal of reversed low-pass
        let g: [f64; 4] = [DB4_H[3], -DB4_H[2], DB4_H[1], -DB4_H[0]];
        for (k, &gk) in g.iter().enumerate() {
            let idx = (2 * i + k) % n;
            detail[i] += gk * data[idx];
        }
    }

    data[..half].copy_from_slice(&approx);
    data[half..].copy_from_slice(&detail);
}

/// Daubechies-4 inverse wavelet transform (one level).
///
/// # Panics
///
/// Panics if `data.len()` is less than 4 or not even.
pub fn db4_inverse(data: &mut [f64]) {
    let n = data.len();
    assert!(
        n >= 4 && n.is_multiple_of(2),
        "length must be >= 4 and even"
    );

    let half = n / 2;
    let approx = data[..half].to_vec();
    let detail = data[half..].to_vec();

    let g: [f64; 4] = [DB4_H[3], -DB4_H[2], DB4_H[1], -DB4_H[0]];
    let mut result = vec![0.0; n];

    for i in 0..half {
        for (k, &hk) in DB4_H.iter().enumerate() {
            let idx = (2 * i + k) % n;
            result[idx] += hk * approx[i];
        }
        for (k, &gk) in g.iter().enumerate() {
            let idx = (2 * i + k) % n;
            result[idx] += gk * detail[i];
        }
    }

    data.copy_from_slice(&result);
}
