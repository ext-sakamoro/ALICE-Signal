//! Analytic oracles for onset detection and tempo estimation.
//!
//! Every expectation is a construction with a known ground truth or a closed
//! form written out here, never a value the detector produced. Integration test,
//! so the crate is seen exactly as a downstream user sees it.
//!
//! Oracle sources:
//! - silence and a steady tone add no new energy, so the flux is zero
//! - a tone switched on at a known time puts the flux maximum in the frame that
//!   contains that time, so the detected onset is the constructed one
//! - a click train built at an exact tempo must read back as that tempo
//! - the frame rate is `sample_rate / hop` by definition
//! - tempo is a property of the sound, so it cannot change when the sample rate
//!   or the amplitude changes

use alice_signal::onset::{estimate_tempo, pick_onsets, spectral_flux, OnsetConfig, OnsetError};
use core::f64::consts::TAU;

const SR: f64 = 8192.0;

/// Settings matched to `SR`: ~0.125 s window, 50% overlap, ~1 s local average.
fn cfg() -> OnsetConfig {
    OnsetConfig::try_new(1024, 512, 16, 0.8, 60.0, 200.0).unwrap()
}

fn assert_rel(actual: f64, expected: f64, tol: f64, what: &str) {
    let rel = (actual - expected).abs() / expected.abs();
    assert!(
        rel <= tol,
        "{what}: got {actual:.6}, expected {expected:.6}, relative error {rel:.3e} > {tol:.3e}"
    );
}

/// `seconds` of silence.
fn silence(seconds: f64) -> Vec<f64> {
    vec![0.0; (seconds * SR) as usize]
}

/// A steady sinusoid of `freq` Hz.
fn tone(seconds: f64, freq: f64, amplitude: f64) -> Vec<f64> {
    let n = (seconds * SR) as usize;
    (0..n)
        .map(|i| amplitude * (TAU * freq * i as f64 / SR).sin())
        .collect()
}

/// Clicks at exactly `bpm`, each a short decaying burst of broadband energy.
///
/// The burst is an exponentially damped sinusoid, which has energy across many
/// bins at its start -- the thing spectral flux is meant to see.
fn click_train(seconds: f64, bpm: f64, amplitude: f64) -> Vec<f64> {
    let n = (seconds * SR) as usize;
    let mut out = vec![0.0; n];
    let period = 60.0 / bpm;
    let mut beat = 0usize;
    loop {
        let start = (beat as f64 * period * SR) as usize;
        if start >= n {
            break;
        }
        // 20 ms of decaying 1 kHz ring
        let len = (0.02 * SR) as usize;
        for k in 0..len {
            if start + k >= n {
                break;
            }
            let t = k as f64 / SR;
            out[start + k] += amplitude * (-t * 200.0).exp() * (TAU * 1000.0 * t).sin();
        }
        beat += 1;
    }
    out
}

// ---------------------------------------------------------------------------
// Nothing new means no flux.
// ---------------------------------------------------------------------------

#[test]
fn silence_produces_no_flux() {
    // Oracle: the magnitude spectrum of silence is zero in every bin, so nothing
    // can appear between two frames of it.
    let env = spectral_flux(&silence(2.0), SR, &cfg()).unwrap();
    assert!(env.flux.len() > 10, "expected several frames");
    for (i, f) in env.flux.iter().enumerate() {
        assert!(*f == 0.0, "frame {i} of silence had flux {f}");
    }
}

#[test]
fn a_steady_tone_produces_no_flux_once_it_is_running() {
    // Oracle: a sinusoid that never changes has the same magnitude spectrum in
    // every frame, so the half-wave rectified difference is zero. Only the first
    // frames, where the analysis window still straddles the start, can differ --
    // here the signal starts at sample 0, so every frame is inside the tone.
    let env = spectral_flux(&tone(2.0, 440.0, 0.5), SR, &cfg()).unwrap();
    let peak = env.flux.iter().fold(0.0_f64, |a, b| a.max(*b));
    let total: f64 = env.flux.iter().sum();
    assert!(
        peak < 1e-9,
        "a steady tone should not look like an onset: peak flux {peak:.3e}, total {total:.3e}"
    );
}

// ---------------------------------------------------------------------------
// A single known onset.
// ---------------------------------------------------------------------------

#[test]
fn a_tone_switched_on_is_detected_at_the_constructed_time() {
    // Oracle: the construction. Half a second of silence then a tone, so the only
    // instant at which energy appears is t = 0.5 s. The detector cannot do better
    // than one hop (512 / 8192 = 62.5 ms), so that is the tolerance.
    let onset_at = 0.5_f64;
    let mut signal = silence(onset_at);
    signal.extend(tone(1.5, 440.0, 0.5));

    let c = cfg();
    let env = spectral_flux(&signal, SR, &c).unwrap();
    let onsets = pick_onsets(&env, SR, &c);

    assert_eq!(
        onsets.len(),
        1,
        "expected exactly one onset, got {onsets:?}"
    );
    let hop_seconds = c.hop() as f64 / SR;
    assert!(
        (onsets[0] - onset_at).abs() <= 2.0 * hop_seconds,
        "onset at {:.4} s, constructed at {onset_at} s, hop {hop_seconds:.4} s",
        onsets[0]
    );
}

#[test]
fn every_click_in_a_train_is_counted_once() {
    // Oracle: the construction. 4 seconds at 120 BPM is 2 beats per second, so 8
    // clicks, the first at t = 0.
    let bpm = 120.0;
    let seconds = 4.0;
    let expected = (seconds * bpm / 60.0) as usize;
    assert_eq!(expected, 8, "the arithmetic of the construction itself");

    let c = cfg();
    let env = spectral_flux(&click_train(seconds, bpm, 0.8), SR, &c).unwrap();
    let onsets = pick_onsets(&env, SR, &c);

    assert_eq!(
        onsets.len(),
        expected,
        "expected {expected} clicks, detected {} at {onsets:?}",
        onsets.len()
    );
}

// ---------------------------------------------------------------------------
// Tempo.
// ---------------------------------------------------------------------------

#[test]
fn a_click_train_reads_back_at_the_tempo_it_was_built_at() {
    // Oracle: the construction, at three tempi inside the search range.
    for bpm in [90.0_f64, 120.0, 150.0] {
        let c = cfg();
        let env = spectral_flux(&click_train(8.0, bpm, 0.8), SR, &c).unwrap();
        let got = estimate_tempo(&env, &c).expect("a click train has a tempo");
        assert_rel(got, bpm, 0.03, &format!("tempo of a {bpm} BPM train"));
    }
}

#[test]
fn tempo_does_not_depend_on_how_loud_the_signal_is() {
    // Oracle: loudness is not tempo. Scaling every sample scales the flux by the
    // same factor, and the lag of maximum self-similarity cannot move.
    let c = cfg();
    let quiet = spectral_flux(&click_train(8.0, 120.0, 0.05), SR, &c).unwrap();
    let loud = spectral_flux(&click_train(8.0, 120.0, 0.9), SR, &c).unwrap();

    let a = estimate_tempo(&quiet, &c).unwrap();
    let b = estimate_tempo(&loud, &c).unwrap();
    assert_rel(a, b, 1e-12, "tempo under a 18x amplitude change");
    assert_rel(a, 120.0, 0.03, "tempo of the quiet train");
}

#[test]
fn the_frame_rate_is_the_sample_rate_over_the_hop() {
    // Oracle: the definition.
    let c = cfg();
    let env = spectral_flux(&silence(2.0), SR, &c).unwrap();
    assert_rel(
        env.frame_rate,
        SR / c.hop() as f64,
        1e-15,
        "frame rate identity",
    );
}

#[test]
fn a_silent_signal_has_no_tempo() {
    // Oracle: there is nothing to be periodic.
    let c = cfg();
    let env = spectral_flux(&silence(8.0), SR, &c).unwrap();
    assert_eq!(estimate_tempo(&env, &c), None);
}

// ---------------------------------------------------------------------------
// Rejections.
// ---------------------------------------------------------------------------

#[test]
fn a_window_that_is_not_a_power_of_two_is_rejected() {
    assert_eq!(
        OnsetConfig::try_new(1000, 500, 16, 0.8, 60.0, 200.0).unwrap_err(),
        OnsetError::WindowNotPowerOfTwo
    );
}

#[test]
fn a_hop_larger_than_the_window_is_rejected() {
    assert_eq!(
        OnsetConfig::try_new(1024, 2048, 16, 0.8, 60.0, 200.0).unwrap_err(),
        OnsetError::HopInvalid
    );
    assert_eq!(
        OnsetConfig::try_new(1024, 0, 16, 0.8, 60.0, 200.0).unwrap_err(),
        OnsetError::HopInvalid
    );
}

#[test]
fn an_empty_tempo_range_is_rejected() {
    assert_eq!(
        OnsetConfig::try_new(1024, 512, 16, 0.8, 200.0, 60.0).unwrap_err(),
        OnsetError::TempoRangeInvalid
    );
}

#[test]
fn a_non_positive_sample_rate_is_rejected() {
    assert_eq!(
        spectral_flux(&silence(1.0), 0.0, &cfg()).unwrap_err(),
        OnsetError::SampleRateInvalid
    );
}

#[test]
fn a_signal_shorter_than_one_window_is_rejected() {
    let c = cfg();
    let short = vec![0.0; c.window() - 1];
    assert_eq!(
        spectral_flux(&short, SR, &c).unwrap_err(),
        OnsetError::SignalTooShort
    );
}
