//! fft.

use crate::complex::*;
use std::f64::consts::PI;

// FFT (Cooley-Tukey radix-2 DIT)
// ---------------------------------------------------------------------------

/// Checks whether `n` is a power of two.
#[must_use]
pub const fn is_power_of_two(n: usize) -> bool {
    n > 0 && n.is_power_of_two()
}

/// In-place Cooley-Tukey radix-2 decimation-in-time FFT.
///
/// # Panics
///
/// Panics if `buf.len()` is not a power of two.
pub fn fft(buf: &mut [Complex]) {
    let n = buf.len();
    assert!(is_power_of_two(n), "FFT length must be a power of two");

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }

    // Butterfly stages
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * PI / len as f64;
        let wn = Complex::from_polar(1.0, angle);

        let mut start = 0;
        while start < n {
            let mut w = Complex::new(1.0, 0.0);
            for k in 0..half {
                let u = buf[start + k];
                let t = w * buf[start + k + half];
                buf[start + k] = u + t;
                buf[start + k + half] = u - t;
                w = w * wn;
            }
            start += len;
        }
        len <<= 1;
    }
}

/// Inverse FFT via conjugation trick.
///
/// # Panics
///
/// Panics if `buf.len()` is not a power of two.
pub fn ifft(buf: &mut [Complex]) {
    let n = buf.len() as f64;
    for c in buf.iter_mut() {
        *c = c.conj();
    }
    fft(buf);
    for c in buf.iter_mut() {
        *c = c.conj() * (1.0 / n);
    }
}
