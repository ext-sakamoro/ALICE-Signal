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

/// The twiddle table of one FFT length, built once and reused for every buffer of that length.
///
/// [`fft`] / [`ifft`] build one per call; a caller that transforms many buffers of the same
/// length (frame by frame) keeps a plan instead. The outputs are bit-identical either way.
#[derive(Debug, Clone)]
pub struct FftPlan {
    /// w_n^k = e^{−2πik/n}, k < n/2.
    twiddles: Vec<Complex>,
    n: usize,
}

impl FftPlan {
    /// The plan for length `n`.
    ///
    /// Each twiddle factor is computed directly from its angle with `alice_det_math` (no
    /// recurrence: a product chain accumulates rounding along a stage, and the platform
    /// sin / cos differ in the last bits across targets).
    ///
    /// # Panics
    ///
    /// Panics if `n` is not a power of two.
    #[must_use]
    pub fn new(n: usize) -> Self {
        assert!(is_power_of_two(n), "FFT length must be a power of two");
        let twiddles = (0..n / 2)
            .map(|k| {
                let (s, c) = alice_det_math::sin_cos64(-2.0 * PI * k as f64 / n as f64);
                Complex::new(c, s)
            })
            .collect();
        Self { twiddles, n }
    }

    /// The length this plan transforms.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.n
    }

    /// Whether the plan is for length 0 (never: a plan's length is a power of two).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.n == 0
    }

    fn check(&self, buf: &[Complex]) {
        assert!(
            buf.len() == self.n,
            "buffer length {} differs from the plan's length {}",
            buf.len(),
            self.n
        );
    }

    /// In-place forward transform.
    ///
    /// # Panics
    ///
    /// Panics if `buf.len()` is not the plan's length.
    pub fn forward(&self, buf: &mut [Complex]) {
        self.check(buf);
        let n = self.n;

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

        // Butterfly stages: stage `len` uses every (n / len)-th twiddle
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let stride = n / len;
            let mut start = 0;
            while start < n {
                for k in 0..half {
                    let u = buf[start + k];
                    let t = self.twiddles[k * stride] * buf[start + k + half];
                    buf[start + k] = u + t;
                    buf[start + k + half] = u - t;
                }
                start += len;
            }
            len <<= 1;
        }
    }

    /// In-place inverse transform (conjugation trick, scaled by `1 / n`).
    ///
    /// # Panics
    ///
    /// Panics if `buf.len()` is not the plan's length.
    pub fn inverse(&self, buf: &mut [Complex]) {
        self.check(buf);
        let n = self.n as f64;
        for c in buf.iter_mut() {
            *c = c.conj();
        }
        self.forward(buf);
        for c in buf.iter_mut() {
            *c = c.conj() * (1.0 / n);
        }
    }
}

/// In-place Cooley-Tukey radix-2 decimation-in-time FFT.
///
/// Builds the twiddle table on every call; see [`FftPlan`] to reuse it.
///
/// # Panics
///
/// Panics if `buf.len()` is not a power of two.
pub fn fft(buf: &mut [Complex]) {
    FftPlan::new(buf.len()).forward(buf);
}

/// Inverse FFT via conjugation trick.
///
/// Builds the twiddle table on every call; see [`FftPlan`] to reuse it.
///
/// # Panics
///
/// Panics if `buf.len()` is not a power of two.
pub fn ifft(buf: &mut [Complex]) {
    FftPlan::new(buf.len()).inverse(buf);
}
