//! `GNSS` (Global Navigation Satellite System) specific signal-processing
//! primitives.
//!
//! Two operations here bridge the generic DSP layer with concrete satellite
//! navigation use cases (SPACID's `QZSS` authenticity checks in particular):
//!
//! - `PRN` code generation for GPS L1 C/A (Gold codes over 10-bit LFSRs),
//!   which are shared by GPS and `QZSS` L1 C/A.
//! - `C/N0` estimation via the Narrowband / Wideband Power Ratio method
//!   (Beaulieu 1995) — a standard receiver-side metric for signal quality
//!   used to detect jamming and spoofing anomalies.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of chips in one full `C/A` code period.
pub const CA_CODE_LENGTH: usize = 1023;
/// Chip rate of the `C/A` code (chips per second).
pub const CA_CHIP_RATE_HZ: f64 = 1_023_000.0;

// ---------------------------------------------------------------------------
// Gold code / C/A code generator
// ---------------------------------------------------------------------------

/// Table of `(G2_tap_a, G2_tap_b)` pairs indexed by PRN number for GPS/QZSS
/// L1 C/A signals. The chip taken from `G2` is the XOR of the two tap
/// positions.
///
/// Values for PRN 1 through 32 follow IS-GPS-200 Table 3-Ia; PRN 193-197
/// (QZSS) are covered by IS-QZSS-PNT.
#[must_use]
pub fn g2_taps(prn: u16) -> Option<(u8, u8)> {
    let taps = match prn {
        1 => (2, 6),
        2 => (3, 7),
        3 => (4, 8),
        4 => (5, 9),
        5 => (1, 9),
        6 => (2, 10),
        7 => (1, 8),
        8 => (2, 9),
        9 => (3, 10),
        10 => (2, 3),
        11 => (3, 4),
        12 => (5, 6),
        13 => (6, 7),
        14 => (7, 8),
        15 => (8, 9),
        16 => (9, 10),
        17 => (1, 4),
        18 => (2, 5),
        19 => (3, 6),
        20 => (4, 7),
        21 => (5, 8),
        22 => (6, 9),
        23 => (1, 3),
        24 => (4, 6),
        25 => (5, 7),
        26 => (6, 8),
        27 => (7, 9),
        28 => (8, 10),
        29 => (1, 6),
        30 => (2, 7),
        31 => (3, 8),
        32 => (4, 9),
        // QZSS (IS-QZSS-PNT Appendix)
        193 => (4, 9),
        194 => (5, 10),
        195 => (4, 10),
        196 => (7, 10),
        197 => (5, 6),
        _ => return None,
    };
    Some(taps)
}

/// Generate one full period of the GPS/QZSS L1 C/A `PRN` code as ±1 samples.
///
/// Returns `None` for PRNs not covered by [`g2_taps`].
///
/// The code is generated from two 10-bit LFSRs:
///
/// - `G1` polynomial `1 + x³ + x¹⁰`
/// - `G2` polynomial `1 + x² + x³ + x⁶ + x⁸ + x⁹ + x¹⁰`
///
/// with both registers initialised to all-ones as required by IS-GPS-200.
#[must_use]
pub fn ca_code(prn: u16) -> Option<Vec<i8>> {
    let (tap_a, tap_b) = g2_taps(prn)?;
    let mut g1: u16 = 0x3FF; // 10 bits, all ones.
    let mut g2: u16 = 0x3FF;
    let mut out = Vec::with_capacity(CA_CODE_LENGTH);
    for _ in 0..CA_CODE_LENGTH {
        // Output is XOR of G1[10] and (G2[a] XOR G2[b]).
        let g1_out = (g1 >> 9) & 1; // bit 10 (index 9).
        let g2_bit_a = (g2 >> (tap_a - 1)) & 1;
        let g2_bit_b = (g2 >> (tap_b - 1)) & 1;
        let g2_out = g2_bit_a ^ g2_bit_b;
        let chip = (g1_out ^ g2_out) as i8;
        // Map {0,1} to {-1,+1} for correlation-friendly output.
        out.push(if chip == 0 { 1 } else { -1 });

        // Advance G1: bits 3 and 10 (indices 2 and 9) feed the new bit 1.
        let new_g1 = ((g1 >> 9) ^ (g1 >> 2)) & 1;
        g1 = ((g1 << 1) | new_g1) & 0x3FF;

        // Advance G2 with taps 2, 3, 6, 8, 9, 10 (indices 1, 2, 5, 7, 8, 9).
        let new_g2 = ((g2 >> 1) ^ (g2 >> 2) ^ (g2 >> 5) ^ (g2 >> 7) ^ (g2 >> 8) ^ (g2 >> 9)) & 1;
        g2 = ((g2 << 1) | new_g2) & 0x3FF;
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// C/N0 estimation (Narrowband/Wideband Power Ratio, Beaulieu 1995)
// ---------------------------------------------------------------------------

/// Configuration for [`estimate_cn0`].
#[derive(Debug, Clone, Copy)]
pub struct Cn0Config {
    /// Number of prompt correlator samples per accumulation window.
    pub samples_per_window: usize,
    /// Number of accumulation windows averaged (must be at least 2).
    pub windows: usize,
    /// Coherent integration time of one prompt sample, in seconds.
    ///
    /// For a standard GPS receiver operating on a 1 ms coherent integration
    /// this is `1e-3`.
    pub coherent_integration_s: f64,
}

impl Default for Cn0Config {
    fn default() -> Self {
        Self {
            samples_per_window: 20,
            windows: 20,
            coherent_integration_s: 1e-3,
        }
    }
}

/// Estimate carrier-to-noise density ratio `C/N0` in dB-Hz from a stream of
/// in-phase / quadrature prompt correlator samples.
///
/// Uses the Narrowband / Wideband Power Ratio method:
///
/// ```text
/// NBP_k = (Σ I)² + (Σ Q)²        (over one window)
/// WBP_k = Σ (I² + Q²)
/// NP    = mean(NBP / WBP)
/// C/N0  = 10 log10( (NP - 1) / (M - NP) ) - 10 log10(T)
/// ```
///
/// where `M = samples_per_window`, `T = coherent_integration_s`.
///
/// # Errors
///
/// Returns `None` when there are not enough samples to form the requested
/// number of windows, when `M <= 1`, or when the estimator falls into a
/// numerically degenerate region.
#[must_use]
pub fn estimate_cn0(i_samples: &[f64], q_samples: &[f64], cfg: Cn0Config) -> Option<f64> {
    if i_samples.len() != q_samples.len() {
        return None;
    }
    if cfg.samples_per_window <= 1 || cfg.windows < 2 {
        return None;
    }
    let needed = cfg.samples_per_window.checked_mul(cfg.windows)?;
    if i_samples.len() < needed {
        return None;
    }

    let m = cfg.samples_per_window;
    let mut np_sum = 0.0_f64;
    let mut np_count = 0.0_f64;

    for k in 0..cfg.windows {
        let start = k * m;
        let end = start + m;
        let mut sum_i = 0.0_f64;
        let mut sum_q = 0.0_f64;
        let mut wbp = 0.0_f64;
        for j in start..end {
            let i_val = i_samples[j];
            let q_val = q_samples[j];
            sum_i += i_val;
            sum_q += q_val;
            wbp += i_val * i_val + q_val * q_val;
        }
        let nbp = sum_i * sum_i + sum_q * sum_q;
        if wbp <= 0.0 {
            return None;
        }
        np_sum += nbp / wbp;
        np_count += 1.0;
    }

    let np = np_sum / np_count;
    let m_f = m as f64;
    // Guard against pathological values that would blow up the log.
    if !(1.0..m_f).contains(&np) {
        return None;
    }
    let ratio = (np - 1.0) / (m_f - np);
    let cn0_db_hz = 10.0 * ratio.log10() - 10.0 * cfg.coherent_integration_s.log10();
    Some(cn0_db_hz)
}

// ---------------------------------------------------------------------------
// Correlation helpers
// ---------------------------------------------------------------------------

/// Compute the normalised cross-correlation between two sequences of ±1
/// samples. The output is in `[-1, 1]`, with `1` for identical sequences.
///
/// Returns `None` when the sequences have different lengths or when the
/// input is empty.
#[must_use]
pub fn normalised_correlation(a: &[i8], b: &[i8]) -> Option<f64> {
    if a.is_empty() || a.len() != b.len() {
        return None;
    }
    let mut sum: i64 = 0;
    for (x, y) in a.iter().zip(b) {
        sum += i64::from(*x) * i64::from(*y);
    }
    Some(sum as f64 / a.len() as f64)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ca_code_prn1_has_correct_length() {
        let code = ca_code(1).unwrap();
        assert_eq!(code.len(), CA_CODE_LENGTH);
    }

    #[test]
    fn ca_code_values_are_bipolar() {
        let code = ca_code(1).unwrap();
        for c in &code {
            assert!(*c == 1 || *c == -1);
        }
    }

    #[test]
    fn ca_code_is_deterministic() {
        let a = ca_code(5).unwrap();
        let b = ca_code(5).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn ca_code_prns_are_distinct() {
        let a = ca_code(1).unwrap();
        let b = ca_code(2).unwrap();
        let corr = normalised_correlation(&a, &b).unwrap();
        // Cross-correlation of distinct Gold codes is small.
        assert!(corr.abs() < 0.1, "|corr| = {}", corr.abs());
    }

    #[test]
    fn ca_code_autocorrelation_is_unity() {
        let a = ca_code(7).unwrap();
        assert!((normalised_correlation(&a, &a).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ca_code_qzss_prn193() {
        let code = ca_code(193).unwrap();
        assert_eq!(code.len(), CA_CODE_LENGTH);
        // Autocorrelation is unity.
        assert!((normalised_correlation(&code, &code).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ca_code_unknown_prn_returns_none() {
        assert!(ca_code(0).is_none());
        assert!(ca_code(200).is_none());
    }

    #[test]
    fn cn0_estimator_flags_high_snr_input() {
        // Build a clean deterministic signal with I = 100 * chip, Q = 0.
        let cfg = Cn0Config::default();
        let n = cfg.samples_per_window * cfg.windows;
        let i_samples: Vec<f64> = (0..n).map(|_| 100.0).collect();
        let q_samples: Vec<f64> = vec![0.0; n];
        // With WBP = M * I², NBP = (M * I)² so NBP/WBP = M -> degenerate.
        // Add tiny noise on Q to move NP away from the M boundary.
        let mut q = q_samples.clone();
        for (idx, val) in q.iter_mut().enumerate() {
            *val = if idx % 2 == 0 { 0.01 } else { -0.01 };
        }
        let cn0 = estimate_cn0(&i_samples, &q, cfg).unwrap();
        // High SNR should give a large positive dB-Hz value.
        assert!(cn0 > 60.0, "cn0 = {cn0}");
    }

    #[test]
    fn cn0_estimator_rejects_short_input() {
        let cfg = Cn0Config::default();
        let short: Vec<f64> = vec![0.0; 10];
        assert!(estimate_cn0(&short, &short, cfg).is_none());
    }

    #[test]
    fn cn0_estimator_rejects_mismatched_lengths() {
        let cfg = Cn0Config::default();
        let a = vec![0.0; 400];
        let b = vec![0.0; 401];
        assert!(estimate_cn0(&a, &b, cfg).is_none());
    }

    #[test]
    fn cn0_estimator_rejects_zero_power() {
        let cfg = Cn0Config::default();
        let n = cfg.samples_per_window * cfg.windows;
        let zero = vec![0.0; n];
        assert!(estimate_cn0(&zero, &zero, cfg).is_none());
    }

    #[test]
    fn normalised_correlation_rejects_length_mismatch() {
        let a = vec![1i8, -1];
        let b = vec![1i8];
        assert!(normalised_correlation(&a, &b).is_none());
    }

    #[test]
    fn normalised_correlation_handles_orthogonal_signals() {
        // Codes with equal counts of ±1 in orthogonal permutation.
        let a = vec![1, -1, 1, -1];
        let b = vec![1, 1, -1, -1];
        let c = normalised_correlation(&a, &b).unwrap();
        assert!(c.abs() < 1e-12);
    }
}
