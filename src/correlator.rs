//! Early / Prompt / Late correlator for `GNSS` code tracking.
//!
//! A `GNSS` receiver locks onto a satellite by continually correlating the
//! received signal with three shifted copies of the local `PRN` code — one
//! aligned (`Prompt`), one advanced by half a chip (`Early`) and one delayed
//! by half a chip (`Late`). The `Early - Late` difference feeds a Delay
//! Lock Loop (`DLL`) discriminator that steers the code numerically
//! controlled oscillator toward perfect alignment.
//!
//! This module implements a discrete-time, half-chip spacing correlator that
//! consumes a `PRN` code from [`crate::gnss`] and an incoming baseband
//! sample buffer. It is designed for `QZSS` L1 C/A monitoring in SPACID's
//! authenticity checks.

use crate::gnss::CA_CODE_LENGTH;

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

/// Output of one correlation window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CorrelatorOutput {
    /// Early accumulator.
    pub early: f64,
    /// Prompt accumulator.
    pub prompt: f64,
    /// Late accumulator.
    pub late: f64,
}

impl CorrelatorOutput {
    /// Non-coherent code discriminator (Early minus Late in envelope).
    /// A positive value means the local code should be advanced.
    #[must_use]
    pub fn early_minus_late(&self) -> f64 {
        self.early.abs() - self.late.abs()
    }

    /// Normalised `E - L` discriminator, in `[-1, 1]` when the prompt
    /// envelope is non-zero.
    #[must_use]
    pub fn normalised_early_minus_late(&self) -> f64 {
        let prompt_abs = self.prompt.abs();
        if prompt_abs < 1e-12 {
            0.0
        } else {
            self.early_minus_late() / prompt_abs
        }
    }

    /// Prompt-envelope power used as a lock indicator.
    #[must_use]
    pub fn prompt_power(&self) -> f64 {
        self.prompt * self.prompt
    }
}

// ---------------------------------------------------------------------------
// Correlator
// ---------------------------------------------------------------------------

/// Discrete-time correlator with `Early`, `Prompt` and `Late` taps at
/// `±half_chip_offset` samples relative to the current alignment.
///
/// Samples are treated as real-valued in-phase measurements; extending to
/// complex (`I + jQ`) inputs is a matter of pairing two instances.
#[derive(Debug, Clone)]
pub struct Correlator {
    prn: Vec<i8>,
    half_chip_offset: usize,
}

impl Correlator {
    /// Construct a correlator around a full-period `PRN` code (typically
    /// `CA_CODE_LENGTH` samples) with the supplied half-chip offset.
    ///
    /// # Panics
    ///
    /// Panics if `prn` is empty or `half_chip_offset == 0`.
    #[must_use]
    pub fn new(prn: Vec<i8>, half_chip_offset: usize) -> Self {
        assert!(!prn.is_empty(), "empty PRN code");
        assert!(half_chip_offset > 0, "offset must be positive");
        Self {
            prn,
            half_chip_offset,
        }
    }

    /// Length of the correlator's local code.
    #[must_use]
    pub fn code_len(&self) -> usize {
        self.prn.len()
    }

    /// Correlate `samples` against the local code starting at `code_offset`.
    ///
    /// Returns [`CorrelatorOutput`] when the sample window is at least as
    /// long as the code + the maximum tap offset; otherwise returns `None`.
    #[must_use]
    pub fn correlate(&self, samples: &[f64], code_offset: usize) -> Option<CorrelatorOutput> {
        let n = self.prn.len();
        let needed = n + 2 * self.half_chip_offset;
        if samples.len() < needed {
            return None;
        }
        let mut early = 0.0_f64;
        let mut prompt = 0.0_f64;
        let mut late = 0.0_f64;
        for i in 0..n {
            let code = f64::from(self.prn[(i + code_offset) % n]);
            let early_sample = samples[i + self.half_chip_offset];
            let prompt_sample = samples[i + self.half_chip_offset];
            let late_sample = samples[i + 2 * self.half_chip_offset];
            early += code * samples[i];
            prompt += code * prompt_sample;
            late += code * late_sample;
            let _ = early_sample;
        }
        Some(CorrelatorOutput {
            early,
            prompt,
            late,
        })
    }
}

/// Convenience: correlator using the standard `C/A` code length.
#[must_use]
pub fn make_ca_correlator(prn_code: Vec<i8>) -> Correlator {
    debug_assert_eq!(prn_code.len(), CA_CODE_LENGTH);
    Correlator::new(prn_code, 1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gnss::ca_code;

    #[test]
    fn correlator_requires_min_sample_length() {
        let corr = Correlator::new(vec![1, -1, 1], 1);
        // Needs at least 5 samples (code_len + 2*offset).
        assert!(corr.correlate(&[1.0; 4], 0).is_none());
        assert!(corr.correlate(&[1.0; 5], 0).is_some());
    }

    #[test]
    fn prompt_maximises_when_code_and_signal_align() {
        // Synthetic signal: replicate the PRN as ±1 samples with extra tail
        // so the Early/Prompt/Late window has room.
        let code = ca_code(1).unwrap();
        let mut samples: Vec<f64> = code.iter().map(|&c| f64::from(c)).collect();
        // Extend the buffer so Early/Late taps have data.
        samples.push(samples[0]);
        samples.push(samples[1]);
        let corr = make_ca_correlator(code);
        let out = corr.correlate(&samples, 0).unwrap();
        // At alignment the prompt sum is code_len (all products = 1).
        assert!((out.prompt - CA_CODE_LENGTH as f64).abs() < 1.0);
    }

    #[test]
    fn misalignment_reduces_prompt() {
        let code = ca_code(1).unwrap();
        let mut samples: Vec<f64> = code.iter().map(|&c| f64::from(c)).collect();
        samples.push(samples[0]);
        samples.push(samples[1]);
        let corr = make_ca_correlator(code);
        let aligned = corr.correlate(&samples, 0).unwrap().prompt.abs();
        let misaligned = corr.correlate(&samples, 100).unwrap().prompt.abs();
        assert!(aligned > misaligned);
    }

    #[test]
    fn normalised_discriminator_is_in_unit_range() {
        let out = CorrelatorOutput {
            early: 10.0,
            prompt: 20.0,
            late: 5.0,
        };
        let disc = out.normalised_early_minus_late();
        assert!(disc.abs() <= 1.0);
    }

    #[test]
    fn zero_prompt_returns_zero_discriminator() {
        let out = CorrelatorOutput {
            early: 5.0,
            prompt: 0.0,
            late: 1.0,
        };
        assert!(out.normalised_early_minus_late().abs() < 1e-12);
    }

    #[test]
    fn prompt_power_matches_square() {
        let out = CorrelatorOutput {
            early: 0.0,
            prompt: 5.0,
            late: 0.0,
        };
        assert!((out.prompt_power() - 25.0).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "empty PRN code")]
    fn empty_prn_panics() {
        let _ = Correlator::new(Vec::new(), 1);
    }

    #[test]
    #[should_panic(expected = "offset must be positive")]
    fn zero_offset_panics() {
        let _ = Correlator::new(vec![1, -1], 0);
    }
}
