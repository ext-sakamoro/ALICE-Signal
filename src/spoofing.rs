//! GNSS spoofing detection primitives.
//!
//! Spoofing is the transmission of counterfeit GNSS signals designed to
//! mislead a receiver about its position, time, or both. Meaconing
//! (replaying real signals with a delay) is a common inexpensive form.
//! Defence-in-depth uses several *cheap* consistency checks that a
//! spoofer must simultaneously defeat:
//!
//! - **Signal strength anomaly** — spoofed signals typically arrive with
//!   `C/N₀` well above the natural range (~35-50 dB-Hz) since a spoofer
//!   overpowers the genuine ~30 mW signal.
//! - **Clock drift consistency** — the receiver's estimated clock bias
//!   should drift monotonically; a sudden jump means a signal-level
//!   attack tried to force a time shift.
//! - **Position jump** — instantaneous 3D position changes greater than
//!   the receiver dynamics allow are physically impossible without a
//!   spoofer or a multipath outburst.
//! - **Satellite count coherence** — real constellations have a
//!   predictable visible-satellite count for the epoch, latitude, and
//!   time-of-day; a sudden loss to "just enough to fix" is suspicious.
//!
//! Each check is a self-contained function returning a normalised
//! anomaly score in `[0, 1]`. The [`Detector`] aggregates them.
//!
//! # References
//!
//! - Humphreys, T. E. (2013), "Detection strategy for cryptographic
//!   GNSS anti-spoofing", IEEE Trans. AES, 49(2), 1073-1090.
//! - Psiaki, M. L. & Humphreys, T. E. (2016), "GNSS spoofing and
//!   detection", Proc. IEEE, 104(6), 1258-1270.
//! - Kaplan, E. D. & Hegarty, C. J. (2017), §16 Interference, Jamming,
//!   and Spoofing.

#![allow(clippy::doc_markdown)]

// ---------------------------------------------------------------------------
// Individual anomaly checks
// ---------------------------------------------------------------------------

/// Score how anomalous the given `C/N₀` value is. Values in `[35, 50]`
/// dB-Hz are considered normal (score 0); values above 55 dB-Hz are
/// almost certainly spoofed (score 1).
#[must_use]
pub fn cn0_anomaly_score(cn0_db_hz: f64) -> f64 {
    if cn0_db_hz.is_nan() {
        return 1.0;
    }
    if cn0_db_hz <= 50.0 {
        0.0
    } else if cn0_db_hz >= 55.0 {
        1.0
    } else {
        (cn0_db_hz - 50.0) / 5.0
    }
}

/// Score how anomalous a receiver clock bias jump is. Values below
/// `1 µs / s` drift are normal; values above `100 µs / s` are almost
/// certainly attacks (score 1).
#[must_use]
pub fn clock_jump_score(delta_bias_s: f64, dt_s: f64) -> f64 {
    if dt_s <= 0.0 {
        return 0.0;
    }
    let drift = delta_bias_s.abs() / dt_s;
    if drift <= 1e-6 {
        0.0
    } else if drift >= 1e-4 {
        1.0
    } else {
        f64::midpoint(drift.log10(), 6.0)
    }
}

/// Score how anomalous a receiver position jump is relative to the
/// declared maximum receiver speed. Under normal conditions the score is
/// 0; above `2× v_max` it is 1.
#[must_use]
pub fn position_jump_score(delta_m: f64, dt_s: f64, v_max_m_s: f64) -> f64 {
    if dt_s <= 0.0 || v_max_m_s <= 0.0 {
        return 0.0;
    }
    let speed = delta_m.abs() / dt_s;
    if speed <= v_max_m_s {
        0.0
    } else if speed >= 2.0 * v_max_m_s {
        1.0
    } else {
        (speed - v_max_m_s) / v_max_m_s
    }
}

/// Score how anomalous the number of tracked satellites is. Fewer than 4
/// is uninformative (score 0); a very large count (> 20) is suspicious
/// because a spoofer may pump extra PRNs to force fix bias.
#[must_use]
pub fn satellite_count_score(count: usize) -> f64 {
    if count <= 20 {
        0.0
    } else if count >= 40 {
        1.0
    } else {
        ((count - 20) as f64) / 20.0
    }
}

// ---------------------------------------------------------------------------
// Detector
// ---------------------------------------------------------------------------

/// Aggregate spoofing detector.
///
/// The detector maintains the last known receiver clock bias and 3D
/// position so it can compare consecutive epochs. Each `push` returns a
/// [`SpoofingReport`] with individual scores and a combined score.
#[derive(Debug, Clone)]
pub struct Detector {
    /// Maximum plausible receiver speed in m/s (default 100 m/s ≈ 360 km/h).
    pub v_max_m_s: f64,
    /// Overall alarm threshold. When the combined score exceeds this
    /// value, `alarm` is set.
    pub alarm_threshold: f64,
    prev_bias_s: Option<f64>,
    prev_pos_ecef_m: Option<[f64; 3]>,
    prev_epoch_s: Option<f64>,
}

impl Default for Detector {
    fn default() -> Self {
        Self {
            v_max_m_s: 100.0,
            alarm_threshold: 0.5,
            prev_bias_s: None,
            prev_pos_ecef_m: None,
            prev_epoch_s: None,
        }
    }
}

impl Detector {
    /// Construct a new detector with the given maximum speed and alarm
    /// threshold.
    #[must_use]
    pub const fn new(v_max_m_s: f64, alarm_threshold: f64) -> Self {
        Self {
            v_max_m_s,
            alarm_threshold,
            prev_bias_s: None,
            prev_pos_ecef_m: None,
            prev_epoch_s: None,
        }
    }

    /// Push a new epoch of observations.
    pub fn push(
        &mut self,
        epoch_s: f64,
        max_cn0_db_hz: f64,
        clock_bias_s: f64,
        pos_ecef_m: [f64; 3],
        tracked_satellites: usize,
    ) -> SpoofingReport {
        let dt = self
            .prev_epoch_s
            .map_or(1.0, |prev| (epoch_s - prev).max(0.0));

        let cn0_score = cn0_anomaly_score(max_cn0_db_hz);
        let clock_score = self
            .prev_bias_s
            .map_or(0.0, |prev| clock_jump_score(clock_bias_s - prev, dt));
        let pos_score = self.prev_pos_ecef_m.map_or(0.0, |prev| {
            let dx = pos_ecef_m[0] - prev[0];
            let dy = pos_ecef_m[1] - prev[1];
            let dz = pos_ecef_m[2] - prev[2];
            let delta = dx.hypot(dy).hypot(dz);
            position_jump_score(delta, dt, self.v_max_m_s)
        });
        let sat_score = satellite_count_score(tracked_satellites);
        // Combined score: max of individual scores (single failure is
        // enough to alarm).
        let combined = cn0_score.max(clock_score).max(pos_score).max(sat_score);

        self.prev_epoch_s = Some(epoch_s);
        self.prev_bias_s = Some(clock_bias_s);
        self.prev_pos_ecef_m = Some(pos_ecef_m);

        SpoofingReport {
            cn0_score,
            clock_score,
            position_score: pos_score,
            satellite_count_score: sat_score,
            combined_score: combined,
            alarm: combined >= self.alarm_threshold,
        }
    }

    /// Reset the detector state.
    pub fn reset(&mut self) {
        self.prev_bias_s = None;
        self.prev_pos_ecef_m = None;
        self.prev_epoch_s = None;
    }
}

// ---------------------------------------------------------------------------
// SpoofingReport
// ---------------------------------------------------------------------------

/// Report emitted by [`Detector::push`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpoofingReport {
    /// C/N₀ anomaly score in `[0, 1]`.
    pub cn0_score: f64,
    /// Clock jump anomaly score in `[0, 1]`.
    pub clock_score: f64,
    /// Position jump anomaly score in `[0, 1]`.
    pub position_score: f64,
    /// Satellite count anomaly score in `[0, 1]`.
    pub satellite_count_score: f64,
    /// Combined score (max of individual scores).
    pub combined_score: f64,
    /// Whether the combined score exceeded the detector's threshold.
    pub alarm: bool,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cn0_within_normal_range_scores_zero() {
        assert!((cn0_anomaly_score(45.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn cn0_above_55_saturates_at_one() {
        assert!((cn0_anomaly_score(60.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cn0_nan_maps_to_maximum_anomaly() {
        assert!((cn0_anomaly_score(f64::NAN) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn slow_clock_drift_scores_zero() {
        // 1 ns over 1 second → 1e-9 drift, well below 1 µs/s.
        assert!((clock_jump_score(1e-9, 1.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn fast_clock_jump_saturates_at_one() {
        // 1 ms in 1 s = 1e-3 drift, far above 1e-4.
        assert!((clock_jump_score(1e-3, 1.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn position_within_dynamic_bounds_scores_zero() {
        // 50 m in 1 s at 100 m/s max → below limit.
        assert!((position_jump_score(50.0, 1.0, 100.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn position_jump_above_2x_max_saturates() {
        // 500 m in 1 s at 100 m/s max → 5× → 1.0
        assert!((position_jump_score(500.0, 1.0, 100.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn satellite_count_20_or_below_scores_zero() {
        assert!((satellite_count_score(18) - 0.0).abs() < 1e-12);
        assert!((satellite_count_score(20) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn satellite_count_beyond_40_saturates() {
        assert!((satellite_count_score(50) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn detector_intact_stream_gives_no_alarm() {
        let mut det = Detector::default();
        let r1 = det.push(0.0, 45.0, 1e-6, [0.0, 0.0, 6_371_000.0], 8);
        let r2 = det.push(1.0, 46.0, 1.001e-6, [1.0, 0.0, 6_371_000.0], 8);
        assert!(!r1.alarm);
        assert!(!r2.alarm);
    }

    #[test]
    fn detector_flags_cn0_spike() {
        let mut det = Detector::default();
        det.push(0.0, 45.0, 0.0, [0.0, 0.0, 6_371_000.0], 8);
        let r = det.push(1.0, 60.0, 1e-8, [1.0, 0.0, 6_371_000.0], 8);
        assert!(r.cn0_score > 0.5);
        assert!(r.alarm);
    }

    #[test]
    fn detector_flags_position_jump() {
        let mut det = Detector::default();
        det.push(0.0, 45.0, 0.0, [0.0, 0.0, 6_371_000.0], 8);
        let r = det.push(1.0, 45.0, 1e-8, [1000.0, 0.0, 6_371_000.0], 8);
        assert!(r.position_score > 0.5);
        assert!(r.alarm);
    }

    #[test]
    fn detector_flags_clock_jump() {
        let mut det = Detector::default();
        det.push(0.0, 45.0, 0.0, [0.0, 0.0, 6_371_000.0], 8);
        let r = det.push(1.0, 45.0, 1e-3, [1.0, 0.0, 6_371_000.0], 8);
        assert!(r.clock_score > 0.5);
        assert!(r.alarm);
    }

    #[test]
    fn reset_clears_previous_state() {
        let mut det = Detector::default();
        det.push(0.0, 45.0, 0.0, [0.0; 3], 8);
        det.reset();
        // Position jump immediately after reset must not fire because
        // there is no previous position.
        let r = det.push(1.0, 45.0, 0.0, [10_000.0, 0.0, 0.0], 8);
        assert!((r.position_score - 0.0).abs() < 1e-12);
    }
}
