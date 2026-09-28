//! Onset detection and tempo estimation from a waveform.
//!
//! The chain is the standard one. Cut the signal into overlapping frames, take
//! the magnitude spectrum of each, and measure how much energy *appeared* since
//! the previous frame — the half-wave rectified spectral flux. A note starting
//! adds energy across many bins at once, so the flux spikes; a steady tone adds
//! nothing, so it does not. Picking the peaks of that curve gives onset times,
//! and the lag that the curve most resembles itself at gives the tempo.
//!
//! The spectrum comes from [`crate::psd::psd_windowed`] rather than a second
//! FFT written here, so there is one Fourier law in this crate, not two.
//!
//! # Octave ambiguity
//!
//! A periodic pulse train correlates with itself at every multiple of its
//! period, so tempo estimation is only ever decided up to a factor of two by the
//! signal alone. [`estimate_tempo`] resolves it by searching a fixed range of
//! plausible tempi and preferring the shorter lag, which is a convention, not a
//! measurement: a 180 BPM piece searched in a 60-200 range can read as 90.

use crate::psd::psd_windowed;
use crate::windows::hanning;

/// Why an onset envelope could not be computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OnsetError {
    /// The analysis window was not a power of two, which the FFT requires.
    WindowNotPowerOfTwo,
    /// The hop was zero, or larger than the window.
    HopInvalid,
    /// A frame span was zero.
    MedianSpanInvalid,
    /// The tempo search range was empty, or not finite and positive.
    TempoRangeInvalid,
    /// The threshold offset was not finite, or was negative.
    ThresholdInvalid,
    /// The relative floor was not finite, or was outside `0.0 ..= 1.0`.
    RelativeFloorInvalid,
    /// The sample rate was not finite and positive.
    SampleRateInvalid,
    /// The signal was shorter than one analysis window.
    SignalTooShort,
}

/// How the waveform is cut up and how peaks are chosen.
///
/// Fields are private and every constructor validates. There is no `Default`:
/// a window length and a hop are an analysis choice with an audible effect on
/// the answer, so picking them silently would hide where a tempo came from.
///
/// Built from outside the crate with [`Self::try_new`] or [`Self::preset_music`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct OnsetConfig {
    window: usize,
    hop: usize,
    median_span: usize,
    threshold_delta: f64,
    relative_floor: f64,
    min_bpm: f64,
    max_bpm: f64,
}

impl OnsetConfig {
    /// Build a configuration, checking every value up front.
    ///
    /// `median_span` is how many frames the local average of the flux is taken
    /// over, and `threshold_delta` how far above that average a frame must rise
    /// to count as an onset, in units of the flux standard deviation.
    ///
    /// `relative_floor` is the second half of the test, and the one that keeps a
    /// steady sound quiet. A threshold built out of the curve's own mean and
    /// spread is self-referential: it finds peaks in *any* curve that is not
    /// exactly flat, including one that only wobbles because the analysis window
    /// lands on a different part of a sustained tone each frame. A peak must
    /// therefore also carry this fraction of the frame's total spectral
    /// magnitude, which separates "energy appeared" from "energy is present":
    /// a note starting contributes an appreciable share of the frame, leakage
    /// contributes about `1e-4` of it. Both quantities scale with the signal, so
    /// the test stays independent of how loud the recording is.
    ///
    /// # Errors
    /// [`OnsetError::WindowNotPowerOfTwo`], [`OnsetError::HopInvalid`],
    /// [`OnsetError::MedianSpanInvalid`], [`OnsetError::ThresholdInvalid`] or
    /// [`OnsetError::TempoRangeInvalid`]; see each variant.
    pub fn try_new(
        window: usize,
        hop: usize,
        median_span: usize,
        threshold_delta: f64,
        relative_floor: f64,
        min_bpm: f64,
        max_bpm: f64,
    ) -> Result<Self, OnsetError> {
        if window == 0 || !window.is_power_of_two() {
            return Err(OnsetError::WindowNotPowerOfTwo);
        }
        if hop == 0 || hop > window {
            return Err(OnsetError::HopInvalid);
        }
        if median_span == 0 {
            return Err(OnsetError::MedianSpanInvalid);
        }
        if !threshold_delta.is_finite() || threshold_delta < 0.0 {
            return Err(OnsetError::ThresholdInvalid);
        }
        if !relative_floor.is_finite() || !(0.0..=1.0).contains(&relative_floor) {
            return Err(OnsetError::RelativeFloorInvalid);
        }
        if !min_bpm.is_finite() || !max_bpm.is_finite() || min_bpm <= 0.0 || max_bpm <= min_bpm {
            return Err(OnsetError::TempoRangeInvalid);
        }
        Ok(Self {
            window,
            hop,
            median_span,
            threshold_delta,
            relative_floor,
            min_bpm,
            max_bpm,
        })
    }

    /// Settings that work on music at ordinary sample rates.
    ///
    /// A 1024-sample window hopped every 128 samples, a local average over
    /// about a second of frames at 44.1 kHz, and a 60-200 BPM search. The hop
    /// sets the time resolution of the envelope, and a tempo search needs
    /// several frames per beat to tell a period from its double.
    ///
    /// # Errors
    /// Cannot fail; the signature stays fallible so callers keep one code path.
    pub fn preset_music() -> Result<Self, OnsetError> {
        Self::try_new(1024, 128, 86, 0.8, 0.01, 60.0, 200.0)
    }

    /// Analysis window length in samples.
    #[must_use]
    pub const fn window(self) -> usize {
        self.window
    }

    /// Hop between consecutive frames in samples.
    #[must_use]
    pub const fn hop(self) -> usize {
        self.hop
    }

    /// Frames the local average of the flux is taken over.
    #[must_use]
    pub const fn median_span(self) -> usize {
        self.median_span
    }

    /// Rise above the local average, in flux standard deviations.
    #[must_use]
    pub const fn threshold_delta(self) -> f64 {
        self.threshold_delta
    }

    /// Share of a frame's total spectral magnitude a peak must also carry.
    #[must_use]
    pub const fn relative_floor(self) -> f64 {
        self.relative_floor
    }

    /// Slowest tempo searched.
    #[must_use]
    pub const fn min_bpm(self) -> f64 {
        self.min_bpm
    }

    /// Fastest tempo searched.
    #[must_use]
    pub const fn max_bpm(self) -> f64 {
        self.max_bpm
    }
}

/// The half-wave rectified spectral flux of a signal, frame by frame.
///
/// Produced by [`spectral_flux`] and read by the caller.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct OnsetEnvelope {
    /// One value per analysis frame. The first is always zero: nothing has
    /// appeared yet when there is no previous frame to compare against.
    pub flux: Vec<f64>,
    /// Total spectral magnitude of each frame, in the same units as `flux`.
    ///
    /// The scale a flux value has to be judged against: it says how much sound
    /// is there, where `flux` says how much of it is new.
    pub magnitude: Vec<f64>,
    /// Frames per second, which is `sample_rate / hop`.
    pub frame_rate: f64,
}

impl OnsetEnvelope {
    /// Time in seconds of the centre of frame `index`.
    #[must_use]
    pub fn frame_time(&self, index: usize, window: usize, sample_rate: f64) -> f64 {
        (index as f64).mul_add(1.0 / self.frame_rate, 0.5 * window as f64 / sample_rate)
    }
}

/// Measure how much spectral energy appears from each frame to the next.
///
/// # Errors
/// [`OnsetError::SampleRateInvalid`] or [`OnsetError::SignalTooShort`].
pub fn spectral_flux(
    signal: &[f64],
    sample_rate: f64,
    cfg: &OnsetConfig,
) -> Result<OnsetEnvelope, OnsetError> {
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(OnsetError::SampleRateInvalid);
    }
    if signal.len() < cfg.window {
        return Err(OnsetError::SignalTooShort);
    }

    let window = hanning(cfg.window);
    let frames = (signal.len() - cfg.window) / cfg.hop + 1;
    let mut flux = Vec::with_capacity(frames);
    let mut totals = Vec::with_capacity(frames);
    let mut previous: Option<Vec<f64>> = None;

    for frame in 0..frames {
        let start = frame * cfg.hop;
        // Power per bin from the one Fourier law this crate has; the square root
        // puts it back in amplitude, which is what flux is conventionally in.
        let magnitude: Vec<f64> = psd_windowed(&signal[start..start + cfg.window], &window)
            .iter()
            .map(|power| power.sqrt())
            .collect();

        let value = previous.as_ref().map_or(0.0, |prev| {
            magnitude
                .iter()
                .zip(prev)
                .map(|(now, before)| (now - before).max(0.0))
                .sum()
        });
        flux.push(value);
        totals.push(magnitude.iter().sum());
        previous = Some(magnitude);
    }

    Ok(OnsetEnvelope {
        flux,
        magnitude: totals,
        frame_rate: sample_rate / cfg.hop as f64,
    })
}

/// Times in seconds of the flux peaks that stand out from their neighbourhood.
#[must_use]
pub fn pick_onsets(envelope: &OnsetEnvelope, sample_rate: f64, cfg: &OnsetConfig) -> Vec<f64> {
    let flux = &envelope.flux;
    let n = flux.len();
    if n < 3 {
        return Vec::new();
    }

    let mean = flux.iter().sum::<f64>() / n as f64;
    let deviation = (flux.iter().map(|f| (f - mean) * (f - mean)).sum::<f64>() / n as f64).sqrt();
    // A curve with no spread has no peaks that stand out, however large it is.
    if deviation <= 0.0 {
        return Vec::new();
    }

    let half = cfg.median_span / 2;
    let mut onsets = Vec::new();
    for i in 1..n - 1 {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(n);
        let local = flux[lo..hi].iter().sum::<f64>() / (hi - lo) as f64;
        let threshold = cfg.threshold_delta.mul_add(deviation, local);
        // The relative floor is what stops the self-referential part of the
        // threshold from finding onsets in a sound that never starts anything.
        let floor = cfg.relative_floor * envelope.magnitude[i];
        // `>=` on the left and `>` on the right so a plateau is reported once.
        if flux[i] > threshold
            && flux[i] >= floor
            && flux[i] >= flux[i - 1]
            && flux[i] > flux[i + 1]
        {
            onsets.push(envelope.frame_time(i, cfg.window, sample_rate));
        }
    }
    onsets
}

/// The tempo in beats per minute the envelope most resembles itself at.
///
/// `None` when the envelope is too short to hold one period of the slowest
/// tempo searched, or when it carries no energy at all.
#[must_use]
pub fn estimate_tempo(envelope: &OnsetEnvelope, cfg: &OnsetConfig) -> Option<f64> {
    let flux = &envelope.flux;
    let n = flux.len();
    if n < 4 {
        return None;
    }

    // Centring the curve first, so that the constant part of the envelope does
    // not make every lag look correlated.
    let mean = flux.iter().sum::<f64>() / n as f64;
    let centred: Vec<f64> = flux.iter().map(|f| f - mean).collect();
    if centred.iter().map(|c| c * c).sum::<f64>() <= 0.0 {
        return None;
    }

    let autocorrelation = |lag: usize| -> f64 {
        centred[lag..]
            .iter()
            .zip(&centred[..n - lag])
            .map(|(a, b)| a * b)
            .sum()
    };

    let min_lag = (60.0 / cfg.max_bpm * envelope.frame_rate).floor().max(1.0) as usize;
    let max_lag = ((60.0 / cfg.min_bpm * envelope.frame_rate).ceil() as usize).min(n / 2);
    if max_lag <= min_lag {
        return None;
    }

    // The sum is deliberately not divided by the overlap length. A periodic
    // envelope correlates with itself at every multiple of its period, and
    // leaving the sums unnormalised makes the shorter lag -- the real one --
    // win, because more terms overlap there.
    let mut best_lag = min_lag;
    let mut best_value = autocorrelation(min_lag);
    for lag in min_lag + 1..=max_lag {
        let value = autocorrelation(lag);
        if value > best_value {
            best_value = value;
            best_lag = lag;
        }
    }
    if best_value <= 0.0 {
        return None;
    }

    // The true period rarely lands on a whole frame, so take the vertex of the
    // parabola through the peak and its two neighbours.
    let refined = if best_lag > min_lag && best_lag < max_lag {
        let before = autocorrelation(best_lag - 1);
        let after = autocorrelation(best_lag + 1);
        let curvature = (before - 2.0 * best_value) + after;
        if curvature.abs() > f64::EPSILON {
            0.5f64.mul_add((before - after) / curvature, best_lag as f64)
        } else {
            best_lag as f64
        }
    } else {
        best_lag as f64
    };

    Some(60.0 * envelope.frame_rate / refined)
}
