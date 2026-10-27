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

    // Twiddle factors w_n^k = e^{−2πik/n}, k < n/2, each computed directly from its angle with
    // `alice_det_math` (no recurrence: a product chain accumulates rounding along a stage, and
    // the platform sin / cos differ in the last bits across targets). Stage `len` uses every
    // (n / len)-th entry.
    let twiddles: Vec<Complex> = (0..n / 2)
        .map(|k| {
            let (s, c) = alice_det_math::sin_cos64(-2.0 * PI * k as f64 / n as f64);
            Complex::new(c, s)
        })
        .collect();

    // Butterfly stages
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let stride = n / len;
        let mut start = 0;
        while start < n {
            for k in 0..half {
                let u = buf[start + k];
                let t = twiddles[k * stride] * buf[start + k + half];
                buf[start + k] = u + t;
                buf[start + k + half] = u - t;
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
