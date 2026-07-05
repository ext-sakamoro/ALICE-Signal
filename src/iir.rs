//! iir.

use std::f64::consts::PI;

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
