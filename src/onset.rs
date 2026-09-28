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
    /// # Errors
    /// [`OnsetError::WindowNotPowerOfTwo`], [`OnsetError::HopInvalid`],
    /// [`OnsetError::MedianSpanInvalid`], [`OnsetError::ThresholdInvalid`] or
    /// [`OnsetError::TempoRangeInvalid`]; see each variant.
    pub fn try_new(
        window: usize,
        hop: usize,
        median_span: usize,
        threshold_delta: f64,
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
        if !min_bpm.is_finite() || !max_bpm.is_finite() || min_bpm <= 0.0 || max_bpm <= min_bpm {
            return Err(OnsetError::TempoRangeInvalid);
        }
        Ok(Self {
            window,
            hop,
            median_span,
            threshold_delta,
            min_bpm,
            max_bpm,
        })
    }

    /// Settings that work on music at ordinary sample rates.
    ///
    /// A 1024-sample window with 50% overlap, a local average over about a
    /// second of frames, and a 60-200 BPM search.
    ///
    /// # Errors
    /// Cannot fail; the signature stays fallible so callers keep one code path.
    pub fn preset_music() -> Result<Self, OnsetError> {
        Self::try_new(1024, 512, 43, 0.8, 60.0, 200.0)
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
    _signal: &[f64],
    _sample_rate: f64,
    _cfg: &OnsetConfig,
) -> Result<OnsetEnvelope, OnsetError> {
    todo!("spectral_flux not implemented")
}

/// Times in seconds of the flux peaks that stand out from their neighbourhood.
#[must_use]
pub fn pick_onsets(_envelope: &OnsetEnvelope, _sample_rate: f64, _cfg: &OnsetConfig) -> Vec<f64> {
    todo!("pick_onsets not implemented")
}

/// The tempo in beats per minute the envelope most resembles itself at.
///
/// `None` when the envelope is too short to hold one period of the slowest
/// tempo searched, or when it carries no energy at all.
#[must_use]
pub fn estimate_tempo(_envelope: &OnsetEnvelope, _cfg: &OnsetConfig) -> Option<f64> {
    todo!("estimate_tempo not implemented")
}
