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

/// Settings matched to `SR`: ~0.125 s window, 15.6 ms hop, ~1 s local average.
///
/// The hop is what sets the time resolution of the envelope, and a tempo search
/// needs several frames per beat to tell a period from its double. At 200 BPM
/// this gives 19 frames per beat; a 62.5 ms hop would give under 5, and a
/// 150 BPM train then reads as 75 because the double lag lines up better with
/// the frame grid than the true one does.
fn cfg() -> OnsetConfig {
    OnsetConfig::try_new(1024, 128, 64, 0.8, 0.01, 60.0, 200.0).unwrap()
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
fn a_steady_tone_is_orders_of_magnitude_quieter_than_a_real_onset() {
    // Oracle: a sustained sinusoid adds no new energy, so its flux is bounded by
    // how much the magnitude spectrum wobbles as the window slides across it --
    // spectral leakage, not a note starting.
    //
    // It is NOT exactly zero. A 440 Hz tone at 8192 Hz has a period of 18.6
    // samples, so a 128-sample hop never lands on a whole number of cycles and
    // the leakage pattern differs slightly every frame. The physical claim that
    // can be asserted is therefore a ratio, not an equality: the loudest thing a
    // steady tone can produce must be far below what starting that same tone
    // produces. Measured against the frame's own spectral magnitude, leakage
    // sits near 1e-4 and an onset near 1.
    let c = cfg();
    let steady = spectral_flux(&tone(2.0, 440.0, 0.5), SR, &c).unwrap();
    let steady_peak = steady
        .flux
        .iter()
        .zip(&steady.magnitude)
        .map(|(f, m)| f / m)
        .fold(0.0_f64, f64::max);

    let mut switched = silence(0.5);
    switched.extend(tone(1.5, 440.0, 0.5));
    let onset = spectral_flux(&switched, SR, &c).unwrap();
    let onset_peak = onset
        .flux
        .iter()
        .zip(&onset.magnitude)
        .map(|(f, m)| if *m > 0.0 { f / m } else { 0.0 })
        .fold(0.0_f64, f64::max);

    assert!(
        steady_peak * 100.0 < onset_peak,
        "a steady tone must be at least 100x quieter than the same tone starting: \
         steady {steady_peak:.3e}, onset {onset_peak:.3e}"
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
    // clicks. A quarter second of silence is put in front because the very first
    // click would otherwise sit in frame 0, which by the documented contract has
    // no flux -- see `an_onset_inside_the_first_frame_cannot_be_seen`.
    let bpm = 120.0;
    let seconds = 4.0;
    let expected = (seconds * bpm / 60.0) as usize;
    assert_eq!(expected, 8, "the arithmetic of the construction itself");

    let c = cfg();
    let mut signal = silence(0.25);
    signal.extend(click_train(seconds, bpm, 0.8));
    let env = spectral_flux(&signal, SR, &c).unwrap();
    let onsets = pick_onsets(&env, SR, &c);

    assert_eq!(
        onsets.len(),
        expected,
        "expected {expected} clicks, detected {} at {onsets:?}",
        onsets.len()
    );
}

#[test]
fn an_onset_inside_the_first_frame_cannot_be_seen() {
    // Oracle: the documented contract of `OnsetEnvelope::flux`. Flux measures
    // what appeared since the previous frame, and frame 0 has no previous frame,
    // so a sound that is already there at t = 0 leaves no trace. This is a real
    // limit of any difference-based detector, not a tuning problem: it is pinned
    // here so that a later change cannot quietly turn it into a spurious onset
    // at the start of every recording.
    let c = cfg();
    let env = spectral_flux(&tone(2.0, 440.0, 0.5), SR, &c).unwrap();
    assert!(
        env.flux[0] == 0.0,
        "frame 0 must carry no flux, got {}",
        env.flux[0]
    );
    assert!(
        pick_onsets(&env, SR, &c).is_empty(),
        "a tone present from t = 0 has no detectable onset"
    );
}

#[test]
fn a_note_change_at_constant_energy_is_still_an_onset() {
    // Oracle: physics, and the reason the flux is half-wave rectified.
    //
    // One tone stops and another of the same amplitude starts at the same
    // instant. The total energy in the frame does not change, so the *signed*
    // difference of the spectra sums to about zero -- the bins that emptied
    // cancel the bins that filled. Only the rectified difference, which keeps
    // what appeared and discards what left, still sees a note beginning.
    //
    // Without rectification this instant is invisible, so this test is what
    // makes `.max(0.0)` in the flux load-bearing.
    let c = cfg();
    let mut signal = tone(1.0, 440.0, 0.5);
    signal.extend(tone(1.0, 660.0, 0.5));
    let env = spectral_flux(&signal, SR, &c).unwrap();

    let change_frame = (1.0 * env.frame_rate) as usize;
    let at_change = env.flux[change_frame - 2..=change_frame + 2]
        .iter()
        .fold(0.0_f64, |a, b| a.max(*b));
    let elsewhere = env.flux[..change_frame - 8]
        .iter()
        .fold(0.0_f64, |a, b| a.max(*b));

    assert!(
        at_change > 50.0 * elsewhere,
        "the note change must stand out from the steady part: {at_change:.3e} vs {elsewhere:.3e}"
    );
    let onsets = pick_onsets(&env, SR, &c);
    assert_eq!(
        onsets.len(),
        1,
        "exactly one note change was constructed, detected {onsets:?}"
    );
    assert!(
        (onsets[0] - 1.0).abs() <= 2.0 * c.hop() as f64 / SR,
        "note change detected at {:.4} s, constructed at 1.0 s",
        onsets[0]
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
        OnsetConfig::try_new(1000, 500, 16, 0.8, 0.01, 60.0, 200.0).unwrap_err(),
        OnsetError::WindowNotPowerOfTwo
    );
}

#[test]
fn a_hop_larger_than_the_window_is_rejected() {
    assert_eq!(
        OnsetConfig::try_new(1024, 2048, 16, 0.8, 0.01, 60.0, 200.0).unwrap_err(),
        OnsetError::HopInvalid
    );
    assert_eq!(
        OnsetConfig::try_new(1024, 0, 16, 0.8, 0.01, 60.0, 200.0).unwrap_err(),
        OnsetError::HopInvalid
    );
}

#[test]
fn an_empty_tempo_range_is_rejected() {
    assert_eq!(
        OnsetConfig::try_new(1024, 512, 16, 0.8, 0.01, 200.0, 60.0).unwrap_err(),
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
