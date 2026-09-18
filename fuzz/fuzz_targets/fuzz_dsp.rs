//! Fuzz: arbitrary f64 sequences (NaN / ±inf / empty / odd length) through the
//! DSP primitives — FFT / IFFT (power-of-two padded), PSD (with and without a
//! window), windowed-sinc FIR + biquad IIR, Haar / db4 wavelets, decimation /
//! interpolation, convolution / correlation, energy / rms. None may panic;
//! finite input must give finite output for the linear operators.
#![no_main]

use alice_signal::complex::Complex;
use alice_signal::convolution::{convolve, correlate};
use alice_signal::decimation::{decimate, interpolate, interpolate_linear};
use alice_signal::fft::{fft, ifft};
use alice_signal::fir::{fir_filter, fir_highpass, fir_lowpass};
use alice_signal::iir::Biquad;
use alice_signal::psd::{psd, psd_windowed};
use alice_signal::utility::{energy, rms, zero_pad_to_power_of_two};
use alice_signal::wavelet::{
    db4_forward, db4_inverse, haar_forward, haar_forward_multi, haar_inverse, haar_inverse_multi,
};
use alice_signal::windows::{blackman, hamming, hanning};
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    samples: Vec<f64>,
    factor: u8,
    order: u8,
    cutoff: f64,
    q: f64,
}

fuzz_target!(|input: Input| {
    let x: Vec<f64> = input.samples.into_iter().take(4096).collect();
    // finite AND inside a range where sums of 4096 terms cannot overflow f64
    let finite = x.iter().all(|v| v.is_finite() && v.abs() < 1e300);

    // FFT on the zero-padded power-of-two length
    let padded = zero_pad_to_power_of_two(&x);
    let mut buf: Vec<Complex> = padded.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft(&mut buf);
    ifft(&mut buf);
    if finite {
        assert!(buf.iter().all(|c| c.re.is_finite() && c.im.is_finite()));
    }
    let p = psd(&padded);
    let _ = psd_windowed(&padded, &hanning(padded.len()));
    assert_eq!(p.len(), padded.len() / 2 + 1);

    let _ = (hamming(x.len()), blackman(x.len()));
    let _ = (energy(&x), rms(&x));

    // filters: cutoff clamped into the documented open interval
    let cutoff = if input.cutoff.is_finite() {
        input.cutoff.abs().fract().clamp(0.01, 0.99)
    } else {
        0.5
    };
    let order = usize::from(input.order % 64) + 1;
    let lp = fir_lowpass(order, cutoff);
    let hp = fir_highpass(order, cutoff);
    let _ = (fir_filter(&x, &lp), fir_filter(&x, &hp));
    let q = if input.q.is_finite() { input.q.abs().clamp(0.05, 50.0) } else { 0.707 };
    for b in [Biquad::lowpass(cutoff, q), Biquad::highpass(cutoff, q), Biquad::bandpass(cutoff, q)] {
        let y = b.filter(&x);
        assert_eq!(y.len(), x.len());
    }

    // wavelets need power-of-two (Haar) / even ≥ 4 (db4) lengths
    if padded.len() >= 2 {
        let mut h = padded.clone();
        haar_forward(&mut h);
        haar_inverse(&mut h);
        haar_forward_multi(&mut h);
        haar_inverse_multi(&mut h);
    }
    if padded.len() >= 4 {
        let mut d = padded.clone();
        db4_forward(&mut d);
        db4_inverse(&mut d);
    }

    let factor = usize::from(input.factor % 8) + 1;
    let _ = (decimate(&x, factor), interpolate(&x, factor), interpolate_linear(&x, factor));
    let tail: Vec<f64> = x.iter().rev().take(64).copied().collect();
    let _ = (convolve(&x, &tail), correlate(&x, &tail));
});
