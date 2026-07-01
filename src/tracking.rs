//! `PLL` / `FLL` tracking loop filters for `GNSS` carrier synchronisation.
//!
//! Implements a discrete second-order loop filter driven by:
//!
//! - A Costas Phase Lock Loop (`PLL`) discriminator `atan(Q / I)` — data-
//!   bit insensitive, suitable for L1 C/A carrier phase tracking.
//! - A Frequency Lock Loop (`FLL`) discriminator based on cross-product
//!   `(I_1 * Q_2 - Q_1 * I_2)` — robust under low `C/N0`.
//!
//! The loop filter coefficients are derived from the noise bandwidth `Bn`
//! and damping factor `ξ`, following Kaplan & Hegarty (2017) §5.

// ---------------------------------------------------------------------------
// Discriminators
// ---------------------------------------------------------------------------

/// Costas `PLL` phase discriminator in radians. Returns `atan(Q / I)` clamped
/// to `±π/2` to remain insensitive to 180° phase flips introduced by the
/// navigation data bit.
#[must_use]
pub fn pll_costas_discriminator(prompt_i: f64, prompt_q: f64) -> f64 {
    if prompt_i.abs() < 1e-15 {
        // Avoid division by zero at exactly π/2 boundary.
        return prompt_q.signum() * core::f64::consts::FRAC_PI_2;
    }
    (prompt_q / prompt_i).atan()
}

/// `FLL` cross-product frequency discriminator in Hz.
///
/// `dt` is the interval between the two prompt samples in seconds.
#[must_use]
pub fn fll_cross_product_discriminator(
    prompt_i_1: f64,
    prompt_q_1: f64,
    prompt_i_2: f64,
    prompt_q_2: f64,
    dt: f64,
) -> f64 {
    if dt <= 0.0 {
        return 0.0;
    }
    let cross = prompt_i_1 * prompt_q_2 - prompt_q_1 * prompt_i_2;
    let dot = prompt_i_1 * prompt_i_2 + prompt_q_1 * prompt_q_2;
    let phase = cross.atan2(dot);
    phase / (2.0 * core::f64::consts::PI * dt)
}

// ---------------------------------------------------------------------------
// Second-order loop filter
// ---------------------------------------------------------------------------

/// Discrete-time second-order loop filter driven by a discriminator error.
///
/// State is the current accumulated frequency correction (Hz). Coefficients
/// are computed from `noise_bandwidth_hz` and `damping_factor` per Kaplan &
/// Hegarty (2017) Table 5.4.
#[derive(Debug, Clone, Copy)]
pub struct SecondOrderLoop {
    /// `Bn * ξ` — first-order coefficient (Hz).
    pub k1: f64,
    /// `Bn²` — second-order coefficient (Hz²).
    pub k2: f64,
    /// Accumulated integral state (Hz).
    pub integrator: f64,
    /// Sample interval (s).
    pub dt: f64,
}

impl SecondOrderLoop {
    /// Construct a loop with noise bandwidth `Bn` (Hz) and damping `ξ`
    /// (typically 0.707 for a critically-damped response).
    #[must_use]
    pub fn new(noise_bandwidth_hz: f64, damping_factor: f64, dt: f64) -> Self {
        // Kaplan & Hegarty §5, Table 5.4: for a 2nd-order loop
        //   ω_n = 8 ξ Bn / (4 ξ² + 1)
        //   k1  = 2 ξ ω_n
        //   k2  = ω_n²
        let omega_n = 8.0 * damping_factor * noise_bandwidth_hz
            / (4.0 * damping_factor * damping_factor + 1.0);
        let k1 = 2.0 * damping_factor * omega_n;
        let k2 = omega_n * omega_n;
        Self {
            k1,
            k2,
            integrator: 0.0,
            dt,
        }
    }

    /// Update the loop with a new discriminator error and return the current
    /// frequency correction command in Hz.
    ///
    /// `error` is the discriminator output in radians (`PLL`) or Hz (`FLL`).
    pub fn update(&mut self, error: f64) -> f64 {
        self.integrator += self.k2 * error * self.dt;
        self.k1 * error + self.integrator
    }

    /// Reset the integral state.
    pub fn reset(&mut self) {
        self.integrator = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pll_discriminator_at_zero_phase() {
        assert!(pll_costas_discriminator(1.0, 0.0).abs() < 1e-12);
    }

    #[test]
    fn pll_discriminator_at_45_degrees() {
        let out = pll_costas_discriminator(1.0, 1.0);
        assert!((out - core::f64::consts::FRAC_PI_4).abs() < 1e-12);
    }

    #[test]
    fn pll_discriminator_reports_negative_phase_for_negative_q() {
        let out = pll_costas_discriminator(1.0, -1.0);
        assert!(out < 0.0);
    }

    #[test]
    fn pll_discriminator_handles_zero_i() {
        let pos = pll_costas_discriminator(0.0, 1.0);
        assert!((pos - core::f64::consts::FRAC_PI_2).abs() < 1e-12);
        let neg = pll_costas_discriminator(0.0, -1.0);
        assert!((neg + core::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn fll_discriminator_at_stationary_input_is_zero() {
        assert!(fll_cross_product_discriminator(1.0, 0.0, 1.0, 0.0, 1e-3).abs() < 1e-12);
    }

    #[test]
    fn fll_discriminator_sign_matches_frequency_offset() {
        // 90° rotation between samples over 1 ms -> ~250 Hz frequency.
        let f = fll_cross_product_discriminator(1.0, 0.0, 0.0, 1.0, 1e-3);
        assert!(f > 0.0);
    }

    #[test]
    fn fll_discriminator_returns_zero_for_nonpositive_dt() {
        assert_eq!(
            fll_cross_product_discriminator(1.0, 0.0, 0.0, 1.0, 0.0),
            0.0
        );
    }

    #[test]
    fn second_order_loop_zero_error_yields_zero_output() {
        let mut loop_ = SecondOrderLoop::new(10.0, 0.707, 1e-3);
        let out = loop_.update(0.0);
        assert!(out.abs() < 1e-12);
    }

    #[test]
    fn second_order_loop_accumulates_integrator() {
        let mut loop_ = SecondOrderLoop::new(10.0, 0.707, 1e-3);
        let a = loop_.update(1.0);
        let b = loop_.update(1.0);
        // Integrator grows with each update, so the second output must be larger.
        assert!(b > a);
    }

    #[test]
    fn reset_clears_integrator() {
        let mut loop_ = SecondOrderLoop::new(10.0, 0.707, 1e-3);
        loop_.update(1.0);
        loop_.update(1.0);
        loop_.reset();
        let out = loop_.update(0.0);
        assert!(out.abs() < 1e-12);
    }
}
