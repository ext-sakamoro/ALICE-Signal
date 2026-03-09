#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::similar_names)]

//! ALICE-Signal: Pure Rust digital signal processing library.
//!
//! Provides FFT (Cooley-Tukey radix-2), FIR filter design (windowed sinc),
//! IIR filter (biquad), wavelet transform (Haar, Daubechies-4),
//! window functions (Hamming, Hanning, Blackman), convolution, correlation,
//! power spectral density, decimation, and interpolation.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Complex number (minimal, no external deps)
// ---------------------------------------------------------------------------

/// Minimal complex number for FFT operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    #[must_use]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[must_use]
    pub fn mag(self) -> f64 {
        self.re.hypot(self.im)
    }

    #[must_use]
    pub fn mag_sq(self) -> f64 {
        self.re.mul_add(self.re, self.im * self.im)
    }

    #[must_use]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    #[must_use]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re.mul_add(rhs.re, -(self.im * rhs.im)),
            im: self.re.mul_add(rhs.im, self.im * rhs.re),
        }
    }
}

impl std::ops::Mul<f64> for Complex {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// IIR biquad filter
// ---------------------------------------------------------------------------

/// Biquad filter coefficients (Direct Form I).
#[derive(Debug, Clone, Copy)]
pub struct Biquad {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
}

impl Biquad {
    /// Design a second-order low-pass biquad filter.
    ///
    /// - `fc`: normalised centre frequency (0..1, 1 = Nyquist)
    /// - `q`: quality factor
    #[must_use]
    pub fn lowpass(fc: f64, q: f64) -> Self {
        let w0 = PI * fc;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        let a0 = 1.0 + alpha;
        Self {
            b0: ((1.0 - cos_w0) / 2.0) / a0,
            b1: (1.0 - cos_w0) / a0,
            b2: ((1.0 - cos_w0) / 2.0) / a0,
            a1: (-2.0 * cos_w0) / a0,
            a2: (1.0 - alpha) / a0,
        }
    }

    /// Design a second-order high-pass biquad filter.
    #[must_use]
    pub fn highpass(fc: f64, q: f64) -> Self {
        let w0 = PI * fc;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        let a0 = 1.0 + alpha;
        Self {
            b0: f64::midpoint(1.0, cos_w0) / a0,
            b1: (-(1.0 + cos_w0)) / a0,
            b2: f64::midpoint(1.0, cos_w0) / a0,
            a1: (-2.0 * cos_w0) / a0,
            a2: (1.0 - alpha) / a0,
        }
    }

    /// Design a second-order band-pass biquad filter.
    #[must_use]
    pub fn bandpass(fc: f64, q: f64) -> Self {
        let w0 = PI * fc;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        let a0 = 1.0 + alpha;
        Self {
            b0: alpha / a0,
            b1: 0.0,
            b2: -alpha / a0,
            a1: (-2.0 * cos_w0) / a0,
            a2: (1.0 - alpha) / a0,
        }
    }

    /// Apply biquad filter to an input signal (Direct Form I).
    #[must_use]
    pub fn filter(&self, input: &[f64]) -> Vec<f64> {
        let n = input.len();
        let mut output = vec![0.0; n];
        let (mut x1, mut x2) = (0.0, 0.0);
        let (mut y1, mut y2) = (0.0, 0.0);
        for i in 0..n {
            let x0 = input[i];
            let y0 = self.b0.mul_add(x0, self.b1.mul_add(x1, self.b2 * x2))
                - self.a1.mul_add(y1, self.a2 * y2);
            output[i] = y0;
            x2 = x1;
            x1 = x0;
            y2 = y1;
            y1 = y0;
        }
        output
    }
}

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Tests (100+)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;
    const EPS_LOOSE: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn assert_approx(a: f64, b: f64, tol: f64) {
        assert!(
            approx_eq(a, b, tol),
            "expected {b}, got {a}, diff = {}",
            (a - b).abs()
        );
    }

    // === Complex ===

    #[test]
    fn complex_add() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        let c = a + b;
        assert_approx(c.re, 4.0, EPS);
        assert_approx(c.im, 6.0, EPS);
    }

    #[test]
    fn complex_sub() {
        let c = Complex::new(5.0, 3.0) - Complex::new(2.0, 1.0);
        assert_approx(c.re, 3.0, EPS);
        assert_approx(c.im, 2.0, EPS);
    }

    #[test]
    fn complex_mul() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        let c = a * b;
        assert_approx(c.re, -5.0, EPS);
        assert_approx(c.im, 10.0, EPS);
    }

    #[test]
    fn complex_mag() {
        let c = Complex::new(3.0, 4.0);
        assert_approx(c.mag(), 5.0, EPS);
    }

    #[test]
    fn complex_mag_sq() {
        let c = Complex::new(3.0, 4.0);
        assert_approx(c.mag_sq(), 25.0, EPS);
    }

    #[test]
    fn complex_conj() {
        let c = Complex::new(1.0, 2.0).conj();
        assert_approx(c.re, 1.0, EPS);
        assert_approx(c.im, -2.0, EPS);
    }

    #[test]
    fn complex_from_polar() {
        let c = Complex::from_polar(1.0, 0.0);
        assert_approx(c.re, 1.0, EPS);
        assert_approx(c.im, 0.0, EPS);
    }

    #[test]
    fn complex_from_polar_pi_half() {
        let c = Complex::from_polar(1.0, PI / 2.0);
        assert_approx(c.re, 0.0, EPS);
        assert_approx(c.im, 1.0, EPS);
    }

    #[test]
    fn complex_mul_scalar() {
        let c = Complex::new(2.0, 3.0) * 2.0;
        assert_approx(c.re, 4.0, EPS);
        assert_approx(c.im, 6.0, EPS);
    }

    // === FFT ===

    #[test]
    fn fft_single_element() {
        let mut buf = [Complex::new(5.0, 0.0)];
        fft(&mut buf);
        assert_approx(buf[0].re, 5.0, EPS);
        assert_approx(buf[0].im, 0.0, EPS);
    }

    #[test]
    fn fft_two_elements() {
        let mut buf = [Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
        fft(&mut buf);
        assert_approx(buf[0].re, 2.0, EPS);
        assert_approx(buf[1].re, 0.0, EPS);
    }

    #[test]
    fn fft_four_constant() {
        let mut buf = [Complex::new(1.0, 0.0); 4];
        fft(&mut buf);
        assert_approx(buf[0].re, 4.0, EPS);
        for i in 1..4 {
            assert_approx(buf[i].mag(), 0.0, EPS);
        }
    }

    #[test]
    fn fft_impulse() {
        let mut buf = vec![Complex::new(0.0, 0.0); 8];
        buf[0] = Complex::new(1.0, 0.0);
        fft(&mut buf);
        for c in &buf {
            assert_approx(c.re, 1.0, EPS);
            assert_approx(c.im, 0.0, EPS);
        }
    }

    #[test]
    fn fft_linearity() {
        let n = 8;
        let a: Vec<Complex> = (0..n).map(|i| Complex::new(i as f64, 0.0)).collect();
        let b: Vec<Complex> = (0..n).map(|i| Complex::new((i * 2) as f64, 0.0)).collect();
        let sum_input: Vec<Complex> = a.iter().zip(&b).map(|(&x, &y)| x + y).collect();

        let mut fa = a;
        let mut fb = b;
        let mut fs = sum_input;
        fft(&mut fa);
        fft(&mut fb);
        fft(&mut fs);

        for i in 0..n {
            let expected = fa[i] + fb[i];
            assert_approx(fs[i].re, expected.re, EPS_LOOSE);
            assert_approx(fs[i].im, expected.im, EPS_LOOSE);
        }
    }

    #[test]
    fn ifft_recovers_original() {
        let original: Vec<Complex> = (0..8)
            .map(|i| Complex::new(i as f64, (i * 3) as f64))
            .collect();
        let mut buf = original.clone();
        fft(&mut buf);
        ifft(&mut buf);
        for (a, b) in buf.iter().zip(&original) {
            assert_approx(a.re, b.re, EPS_LOOSE);
            assert_approx(a.im, b.im, EPS_LOOSE);
        }
    }

    #[test]
    fn fft_parseval_theorem() {
        let mut buf: Vec<Complex> = (0..8).map(|i| Complex::new(i as f64, 0.0)).collect();
        let time_energy: f64 = buf.iter().map(|c| c.mag_sq()).sum();
        fft(&mut buf);
        let freq_energy: f64 = buf.iter().map(|c| c.mag_sq()).sum();
        assert_approx(freq_energy, time_energy * 8.0, EPS_LOOSE);
    }

    #[test]
    fn fft_16_point() {
        let n = 16;
        let mut buf: Vec<Complex> = (0..n).map(|i| Complex::new(i as f64, 0.0)).collect();
        let original = buf.clone();
        fft(&mut buf);
        ifft(&mut buf);
        for (a, b) in buf.iter().zip(&original) {
            assert_approx(a.re, b.re, EPS_LOOSE);
        }
    }

    #[test]
    fn fft_pure_sine() {
        let n = 64;
        let freq = 4.0;
        let mut buf: Vec<Complex> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                Complex::new((2.0 * PI * freq * t).sin(), 0.0)
            })
            .collect();
        fft(&mut buf);
        // Peak should be at bin 4 and n-4
        let peak = buf[4].mag();
        assert!(peak > 10.0, "expected peak at bin 4, got {peak}");
    }

    #[test]
    #[should_panic]
    fn fft_non_power_of_two_panics() {
        let mut buf = vec![Complex::new(0.0, 0.0); 5];
        fft(&mut buf);
    }

    // === Window functions ===

    #[test]
    fn hamming_symmetry() {
        let w = hamming(16);
        for i in 0..8 {
            assert_approx(w[i], w[15 - i], EPS);
        }
    }

    #[test]
    fn hamming_length() {
        assert_eq!(hamming(32).len(), 32);
    }

    #[test]
    fn hamming_edges() {
        let w = hamming(16);
        assert!(w[0] < w[8], "edges should be less than center");
    }

    #[test]
    fn hamming_singleton() {
        assert_eq!(hamming(1), vec![1.0]);
    }

    #[test]
    fn hamming_empty() {
        assert!(hamming(0).is_empty());
    }

    #[test]
    fn hanning_symmetry() {
        let w = hanning(16);
        for i in 0..8 {
            assert_approx(w[i], w[15 - i], EPS);
        }
    }

    #[test]
    fn hanning_endpoints_zero() {
        let w = hanning(16);
        assert_approx(w[0], 0.0, EPS);
        assert_approx(w[15], 0.0, EPS);
    }

    #[test]
    fn hanning_length() {
        assert_eq!(hanning(64).len(), 64);
    }

    #[test]
    fn blackman_symmetry() {
        let w = blackman(32);
        for i in 0..16 {
            assert_approx(w[i], w[31 - i], EPS);
        }
    }

    #[test]
    fn blackman_length() {
        assert_eq!(blackman(128).len(), 128);
    }

    #[test]
    fn blackman_center_peak() {
        let w = blackman(33);
        let center = w[16];
        assert_approx(center, 1.0, EPS);
    }

    #[test]
    fn blackman_near_zero_edges() {
        let w = blackman(64);
        assert!(w[0].abs() < 0.01);
    }

    // === FIR ===

    #[test]
    fn fir_lowpass_dc_gain_one() {
        let h = fir_lowpass(30, 0.5);
        let sum: f64 = h.iter().sum();
        assert_approx(sum, 1.0, EPS_LOOSE);
    }

    #[test]
    fn fir_lowpass_symmetry() {
        let h = fir_lowpass(30, 0.3);
        let n = h.len();
        for i in 0..n / 2 {
            assert_approx(h[i], h[n - 1 - i], EPS_LOOSE);
        }
    }

    #[test]
    fn fir_lowpass_tap_count() {
        let h = fir_lowpass(20, 0.4);
        assert_eq!(h.len(), 21);
    }

    #[test]
    fn fir_highpass_dc_zero() {
        let h = fir_highpass(30, 0.3);
        let sum: f64 = h.iter().sum();
        assert_approx(sum, 0.0, EPS_LOOSE);
    }

    #[test]
    fn fir_filter_impulse_response() {
        let coeffs = vec![1.0, 0.5, 0.25];
        let impulse = vec![1.0, 0.0, 0.0, 0.0];
        let out = fir_filter(&impulse, &coeffs);
        assert_approx(out[0], 1.0, EPS);
        assert_approx(out[1], 0.5, EPS);
        assert_approx(out[2], 0.25, EPS);
    }

    #[test]
    #[should_panic]
    fn fir_lowpass_invalid_cutoff_panics() {
        let _ = fir_lowpass(10, 1.5);
    }

    #[test]
    fn fir_lowpass_attenuates_high_freq() {
        let h = fir_lowpass(64, 0.1);
        let n = 256;
        // High-frequency signal (0.45 normalised)
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * 0.45 * i as f64).sin()).collect();
        let out = fir_filter(&signal, &h);
        let out_energy: f64 = out[65..].iter().map(|x| x * x).sum();
        let in_energy: f64 = signal.iter().map(|x| x * x).sum();
        assert!(
            out_energy < in_energy * 0.05,
            "high freq should be attenuated"
        );
    }

    // === IIR biquad ===

    #[test]
    fn biquad_lowpass_dc_passthrough() {
        let bq = Biquad::lowpass(0.5, 0.707);
        let dc: Vec<f64> = vec![1.0; 100];
        let out = bq.filter(&dc);
        assert_approx(*out.last().unwrap(), 1.0, 0.01);
    }

    #[test]
    fn biquad_lowpass_attenuates_high() {
        let bq = Biquad::lowpass(0.1, 0.707);
        let n = 500;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * 0.45 * i as f64).sin()).collect();
        let out = bq.filter(&signal);
        let in_rms = rms(&signal);
        let out_rms = rms(&out[100..]);
        assert!(out_rms < in_rms * 0.3, "should attenuate high freq");
    }

    #[test]
    fn biquad_highpass_blocks_dc() {
        let bq = Biquad::highpass(0.1, 0.707);
        let dc = vec![1.0; 200];
        let out = bq.filter(&dc);
        assert!(out.last().unwrap().abs() < 0.01, "DC should be blocked");
    }

    #[test]
    fn biquad_bandpass_center() {
        // fc=0.25 means center at w0 = PI*0.25, so the signal frequency
        // should be 0.125 cycles/sample (= 0.25 * Nyquist / 2)
        let bq = Biquad::bandpass(0.25, 5.0);
        let n = 1024;
        let signal: Vec<f64> = (0..n).map(|i| (PI * 0.25 * i as f64).sin()).collect();
        let out = bq.filter(&signal);
        let out_rms = rms(&out[200..]);
        assert!(out_rms > 0.1, "center frequency should pass");
    }

    #[test]
    fn biquad_filter_length() {
        let bq = Biquad::lowpass(0.3, 0.707);
        let input = vec![1.0; 50];
        let out = bq.filter(&input);
        assert_eq!(out.len(), 50);
    }

    #[test]
    fn biquad_lowpass_impulse_response() {
        let bq = Biquad::lowpass(0.3, 0.707);
        let mut impulse = vec![0.0; 32];
        impulse[0] = 1.0;
        let out = bq.filter(&impulse);
        assert!(out[0].abs() > 0.0);
    }

    // === Convolution ===

    #[test]
    fn convolve_identity() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0];
        let c = convolve(&a, &b);
        assert_eq!(c, a);
    }

    #[test]
    fn convolve_length() {
        let a = vec![1.0; 5];
        let b = vec![1.0; 3];
        assert_eq!(convolve(&a, &b).len(), 7);
    }

    #[test]
    fn convolve_known() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.5, 1.0];
        let c = convolve(&a, &b);
        assert_approx(c[0], 0.5, EPS);
        assert_approx(c[1], 2.0, EPS);
        assert_approx(c[2], 3.5, EPS);
        assert_approx(c[3], 3.0, EPS);
    }

    #[test]
    fn convolve_commutative() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        let ab = convolve(&a, &b);
        let ba = convolve(&b, &a);
        for (x, y) in ab.iter().zip(&ba) {
            assert_approx(*x, *y, EPS);
        }
    }

    #[test]
    fn convolve_empty() {
        assert!(convolve(&[], &[1.0]).is_empty());
    }

    // === Correlation ===

    #[test]
    fn autocorrelation_peak_at_center() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let ac = correlate(&x, &x);
        let center = x.len() - 1;
        for (i, val) in ac.iter().enumerate() {
            assert!(
                *val <= ac[center] + EPS,
                "autocorrelation peak should be at center, but ac[{i}]={val} > ac[{center}]={}",
                ac[center]
            );
        }
    }

    #[test]
    fn correlation_length() {
        let a = vec![1.0; 5];
        let b = vec![1.0; 3];
        assert_eq!(correlate(&a, &b).len(), 7);
    }

    #[test]
    fn correlation_empty() {
        assert!(correlate(&[], &[1.0]).is_empty());
    }

    #[test]
    fn correlation_known_values() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0, 1.0];
        let c = correlate(&a, &b);
        // Peak at lag corresponding to shift
        assert_approx(c[0], 1.0, EPS);
    }

    // === PSD ===

    #[test]
    fn psd_dc_signal() {
        let signal = vec![1.0; 8];
        let p = psd(&signal);
        assert!(p[0] > 1.0, "DC component should be large");
        for &val in &p[1..] {
            assert_approx(val, 0.0, EPS_LOOSE);
        }
    }

    #[test]
    fn psd_length() {
        let signal = vec![0.0; 16];
        let p = psd(&signal);
        assert_eq!(p.len(), 9); // N/2 + 1
    }

    #[test]
    fn psd_non_negative() {
        let signal: Vec<f64> = (0..32).map(|i| (i as f64).sin()).collect();
        let p = psd(&signal);
        for &val in &p {
            assert!(val >= 0.0, "PSD must be non-negative");
        }
    }

    #[test]
    fn psd_windowed_length() {
        let signal = vec![1.0; 16];
        let window = hamming(16);
        let p = psd_windowed(&signal, &window);
        assert_eq!(p.len(), 9);
    }

    #[test]
    #[should_panic]
    fn psd_non_power_of_two_panics() {
        let _ = psd(&[1.0; 7]);
    }

    // === Haar wavelet ===

    #[test]
    fn haar_roundtrip() {
        let original = vec![1.0, 4.0, -3.0, 0.0];
        let mut data = original.clone();
        haar_forward(&mut data);
        haar_inverse(&mut data);
        for (a, b) in data.iter().zip(&original) {
            assert_approx(*a, *b, EPS_LOOSE);
        }
    }

    #[test]
    fn haar_forward_known() {
        let mut data = vec![1.0, 1.0];
        haar_forward(&mut data);
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        assert_approx(data[0], 2.0 * inv_sqrt2, EPS);
        assert_approx(data[1], 0.0, EPS);
    }

    #[test]
    fn haar_multi_roundtrip() {
        let original = vec![1.0, 4.0, -3.0, 0.0, 2.0, 5.0, -1.0, 3.0];
        let mut data = original.clone();
        haar_forward_multi(&mut data);
        haar_inverse_multi(&mut data);
        for (a, b) in data.iter().zip(&original) {
            assert_approx(*a, *b, EPS_LOOSE);
        }
    }

    #[test]
    fn haar_energy_preservation() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let e_before = energy(&data);
        haar_forward(&mut data);
        let e_after = energy(&data);
        assert_approx(e_before, e_after, EPS_LOOSE);
    }

    #[test]
    #[should_panic]
    fn haar_forward_odd_length_panics() {
        haar_forward(&mut [1.0, 2.0, 3.0]);
    }

    #[test]
    fn haar_forward_4() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let mut data = original.clone();
        haar_forward(&mut data);
        // Verify different from original
        assert!((data[0] - original[0]).abs() > EPS);
    }

    // === DB4 wavelet ===

    #[test]
    fn db4_roundtrip() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data = original.clone();
        db4_forward(&mut data);
        db4_inverse(&mut data);
        for (a, b) in data.iter().zip(&original) {
            assert_approx(*a, *b, EPS_LOOSE);
        }
    }

    #[test]
    fn db4_energy_preservation() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let e_before = energy(&data);
        db4_forward(&mut data);
        let e_after = energy(&data);
        assert_approx(e_before, e_after, EPS_LOOSE);
    }

    #[test]
    fn db4_forward_changes_data() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data = original.clone();
        db4_forward(&mut data);
        assert_ne!(data, original);
    }

    #[test]
    #[should_panic]
    fn db4_too_short_panics() {
        db4_forward(&mut [1.0, 2.0]);
    }

    // === Decimation & interpolation ===

    #[test]
    fn decimate_by_two() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let d = decimate(&x, 2);
        assert_eq!(d, vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn decimate_by_one() {
        let x = vec![1.0, 2.0, 3.0];
        assert_eq!(decimate(&x, 1), x);
    }

    #[test]
    fn decimate_by_three() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let d = decimate(&x, 3);
        assert_eq!(d, vec![1.0, 4.0, 7.0]);
    }

    #[test]
    #[should_panic]
    fn decimate_zero_panics() {
        let _ = decimate(&[1.0], 0);
    }

    #[test]
    fn interpolate_by_two() {
        let x = vec![1.0, 2.0, 3.0];
        let y = interpolate(&x, 2);
        assert_eq!(y.len(), 6);
        assert_approx(y[0], 1.0, EPS);
        assert_approx(y[1], 0.0, EPS);
        assert_approx(y[2], 2.0, EPS);
    }

    #[test]
    fn interpolate_by_one() {
        let x = vec![1.0, 2.0];
        assert_eq!(interpolate(&x, 1), x);
    }

    #[test]
    #[should_panic]
    fn interpolate_zero_panics() {
        let _ = interpolate(&[1.0], 0);
    }

    #[test]
    fn interpolate_linear_known() {
        let x = vec![0.0, 10.0];
        let y = interpolate_linear(&x, 5);
        assert_eq!(y.len(), 6);
        assert_approx(y[0], 0.0, EPS);
        assert_approx(y[1], 2.0, EPS);
        assert_approx(y[2], 4.0, EPS);
        assert_approx(y[5], 10.0, EPS);
    }

    #[test]
    fn interpolate_linear_single() {
        let x = vec![5.0];
        let y = interpolate_linear(&x, 3);
        assert_eq!(y, vec![5.0]);
    }

    #[test]
    fn interpolate_linear_by_one() {
        let x = vec![1.0, 2.0, 3.0];
        let y = interpolate_linear(&x, 1);
        assert_eq!(y, x);
    }

    // === Utility ===

    #[test]
    fn energy_zero() {
        assert_approx(energy(&[0.0; 10]), 0.0, EPS);
    }

    #[test]
    fn energy_known() {
        assert_approx(energy(&[3.0, 4.0]), 25.0, EPS);
    }

    #[test]
    fn rms_constant() {
        assert_approx(rms(&[5.0; 100]), 5.0, EPS);
    }

    #[test]
    fn rms_empty() {
        assert_approx(rms(&[]), 0.0, EPS);
    }

    #[test]
    fn rms_sine_wave() {
        let n = 10000;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * i as f64 / n as f64).sin())
            .collect();
        let r = rms(&signal);
        assert_approx(r, 1.0 / 2.0_f64.sqrt(), 0.01);
    }

    #[test]
    fn zero_pad_power_of_two() {
        let x = vec![1.0, 2.0, 3.0];
        let y = zero_pad_to_power_of_two(&x);
        assert_eq!(y.len(), 4);
        assert_approx(y[0], 1.0, EPS);
        assert_approx(y[3], 0.0, EPS);
    }

    #[test]
    fn zero_pad_already_power() {
        let x = vec![1.0; 8];
        let y = zero_pad_to_power_of_two(&x);
        assert_eq!(y.len(), 8);
    }

    #[test]
    fn zero_pad_empty() {
        let y = zero_pad_to_power_of_two(&[]);
        assert_eq!(y.len(), 1);
    }

    #[test]
    fn is_power_of_two_true() {
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(is_power_of_two(64));
        assert!(is_power_of_two(1024));
    }

    #[test]
    fn is_power_of_two_false() {
        assert!(!is_power_of_two(0));
        assert!(!is_power_of_two(3));
        assert!(!is_power_of_two(5));
        assert!(!is_power_of_two(100));
    }

    // === Integration / round-trip tests ===

    #[test]
    fn fft_convolve_matches_direct() {
        let a = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0];
        let direct = convolve(&a[..3], &b[..3]);

        let mut fa: Vec<Complex> = a.iter().map(|&x| Complex::new(x, 0.0)).collect();
        let mut fb: Vec<Complex> = b.iter().map(|&x| Complex::new(x, 0.0)).collect();
        fft(&mut fa);
        fft(&mut fb);
        let mut fc: Vec<Complex> = fa.iter().zip(&fb).map(|(&a, &b)| a * b).collect();
        ifft(&mut fc);

        for i in 0..direct.len() {
            assert_approx(fc[i].re, direct[i], EPS_LOOSE);
        }
    }

    #[test]
    fn window_applied_reduces_spectral_leakage() {
        let n = 64;
        // Use non-integer bin frequency to cause spectral leakage
        let freq = 4.5;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / n as f64).sin())
            .collect();
        let p_rect = psd(&signal);
        let window = hanning(n);
        let p_win = psd_windowed(&signal, &window);
        // Windowed PSD should have less leakage in far-off bins
        let leakage_rect: f64 = p_rect[15..].iter().sum();
        let leakage_win: f64 = p_win[15..].iter().sum();
        assert!(
            leakage_win < leakage_rect,
            "window should reduce spectral leakage"
        );
    }

    #[test]
    fn decimate_then_interpolate_preserves_samples() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let d = decimate(&x, 2);
        let y = interpolate(&d, 2);
        // Original samples at even indices should survive
        for (i, &v) in d.iter().enumerate() {
            assert_approx(y[i * 2], v, EPS);
        }
    }

    #[test]
    fn fir_then_decimate_anti_aliasing() {
        // Verify FIR lowpass + decimate workflow runs without error
        let n = 256;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                (2.0 * PI * 0.05 * t).sin() + (2.0 * PI * 0.45 * t).sin()
            })
            .collect();
        let h = fir_lowpass(32, 0.5);
        let filtered = fir_filter(&signal, &h);
        let decimated = decimate(&filtered, 2);
        assert_eq!(decimated.len(), (n + h.len() - 1) / 2);
    }

    #[test]
    fn biquad_cascade() {
        let bq1 = Biquad::lowpass(0.3, 0.707);
        let bq2 = Biquad::lowpass(0.3, 0.707);
        let signal: Vec<f64> = (0..128)
            .map(|i| (2.0 * PI * 0.4 * i as f64).sin())
            .collect();
        let stage1 = bq1.filter(&signal);
        let stage2 = bq2.filter(&stage1);
        let rms_in = rms(&signal);
        let rms_out = rms(&stage2[50..]);
        assert!(
            rms_out < rms_in * 0.2,
            "cascaded lowpass should strongly attenuate"
        );
    }

    #[test]
    fn haar_sparse_representation() {
        // Piecewise constant signal should have few non-zero detail coefficients
        let mut data = vec![5.0; 4];
        data.extend_from_slice(&[10.0; 4]);
        let original = data.clone();
        haar_forward_multi(&mut data);
        // Count near-zero coefficients
        let near_zero = data.iter().filter(|&&x| x.abs() < EPS_LOOSE).count();
        assert!(
            near_zero >= 4,
            "piecewise constant should be sparse in wavelet domain"
        );
        haar_inverse_multi(&mut data);
        for (a, b) in data.iter().zip(&original) {
            assert_approx(*a, *b, EPS_LOOSE);
        }
    }

    #[test]
    fn psd_sine_peak_location() {
        let n = 128;
        let bin = 8;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * bin as f64 * i as f64 / n as f64).sin())
            .collect();
        let p = psd(&signal);
        // The peak should be at the frequency bin
        let max_idx = p
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, bin);
    }

    #[test]
    fn fir_bandpass_combination() {
        // Low-pass then high-pass = bandpass effect
        let h_lp = fir_lowpass(32, 0.6);
        let h_hp = fir_highpass(32, 0.2);
        let n = 512;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                (2.0 * PI * 0.1 * t).sin()
                    + (2.0 * PI * 0.3 * t).sin()
                    + (2.0 * PI * 0.45 * t).sin()
            })
            .collect();
        let after_lp = fir_filter(&signal, &h_lp);
        let after_bp = fir_filter(&after_lp, &h_hp);
        assert!(!after_bp.is_empty());
    }

    #[test]
    fn convolve_associative() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 1.0];
        let c = vec![1.0, 0.5];
        let ab_c = convolve(&convolve(&a, &b), &c);
        let a_bc = convolve(&a, &convolve(&b, &c));
        for (x, y) in ab_c.iter().zip(&a_bc) {
            assert_approx(*x, *y, EPS_LOOSE);
        }
    }

    #[test]
    fn db4_multi_level() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let original = data.clone();
        db4_forward(&mut data);
        // Only one level - just verify roundtrip
        db4_inverse(&mut data);
        for (a, b) in data.iter().zip(&original) {
            assert_approx(*a, *b, EPS_LOOSE);
        }
    }

    #[test]
    fn interpolate_linear_preserves_endpoints() {
        let x = vec![0.0, 10.0, 5.0];
        let y = interpolate_linear(&x, 4);
        assert_approx(y[0], 0.0, EPS);
        assert_approx(*y.last().unwrap(), 5.0, EPS);
    }

    #[test]
    fn hamming_range() {
        let w = hamming(64);
        for &v in &w {
            assert!(v >= 0.0 && v <= 1.0, "hamming values should be in [0,1]");
        }
    }

    #[test]
    fn hanning_range() {
        let w = hanning(64);
        for &v in &w {
            assert!(v >= 0.0 && v <= 1.0, "hanning values should be in [0,1]");
        }
    }

    #[test]
    fn blackman_range() {
        let w = blackman(64);
        for &v in &w {
            assert!(
                v >= -0.01 && v <= 1.01,
                "blackman values should be approximately in [0,1]"
            );
        }
    }
}
