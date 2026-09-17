//! Analytic oracles — closed-form checks for the DSP / GNSS laws in
//! ALICE-Signal (CLAUDE.md § 解析解突合テスト規律, 2026-09-17).
//!
//! Expected values come from closed forms, published tables or an independent
//! reference written in this file (naive O(N²) DFT), never from the crate
//! function under test.
//!
//! Oracle sources:
//! - FFT: naive DFT definition, Parseval Σ|x|² = (1/N)Σ|X|², a tone of
//!   amplitude A on bin k has |X_k| = A·N/2, ifft∘fft = id
//! - windows: closed-form sums (Hann (n−1)/2, Hamming 0.54n−0.46,
//!   Blackman 0.42(n−1)), endpoints and centre values
//! - FIR windowed sinc: DC gain 1, −6 dB at the cutoff, Hamming stopband
//!   ≤ −50 dB; spectral inversion swaps DC and Nyquist
//! - RBJ biquad: Butterworth Q = 1/√2 gives |H(fc)| = 1/√2 exactly, DC /
//!   Nyquist gains 1 / 0 (low-pass), 0 / 1 (high-pass), band-pass peak 1
//! - periodogram: a one-sided PSD carries the full power, Σ psd = Σx²,
//!   tone on bin k ⇒ psd[k] = A²N/2
//! - decimation / interpolation: bin index scales with the rate, linear
//!   interpolation reproduces a linear function exactly
//! - convolution: δ identity, box⊛box = triangle, correlate(a,b) =
//!   convolve(a, reverse b), zero-lag autocorrelation = energy
//! - wavelets: orthonormal (Parseval + perfect reconstruction), a constant
//!   has zero detail and approximation c·√2 (Σh = √2)
//! - Kalman 1-D: Riccati fixed point P⁻ = (q + √(q² + 4qr))/2
//! - tracking: atan discriminators, Kaplan & Hegarty Bn ↔ ω_n inverse
//!   formula, loop step response k1·e + k2·e·n·dt
//! - GNSS: IS-GPS-200 first-10-chip octals for PRN 1-10, Gold code
//!   three-valued correlation {−65, −1, 63}/1023, balance 512/511,
//!   NWPR C/N0 = 10 log10(A²/(2σ²)) − 10 log10(T)

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop
)]

use std::f64::consts::{PI, SQRT_2, TAU};

use alice_signal::complex::Complex;
use alice_signal::convolution::{convolve, correlate};
use alice_signal::decimation::{decimate, interpolate, interpolate_linear};
use alice_signal::fft::{fft, ifft};
use alice_signal::fir::{fir_filter, fir_highpass, fir_lowpass};
use alice_signal::gnss::{
    ca_code, estimate_cn0, normalised_correlation, Cn0Config, CA_CODE_LENGTH,
};
use alice_signal::iir::Biquad;
use alice_signal::kalman::KalmanFilter;
use alice_signal::psd::{psd, psd_windowed};
use alice_signal::tracking::{
    fll_cross_product_discriminator, pll_costas_discriminator, SecondOrderLoop,
};
use alice_signal::utility::{energy, rms, zero_pad_to_power_of_two};
use alice_signal::wavelet::{
    db4_forward, db4_inverse, haar_forward, haar_forward_multi, haar_inverse, haar_inverse_multi,
};
use alice_signal::windows::{blackman, hamming, hanning};

/// Independent reference: the DFT by its definition, O(N²).
fn naive_dft(x: &[Complex]) -> Vec<Complex> {
    let n = x.len();
    (0..n)
        .map(|k| {
            let mut acc = Complex::new(0.0, 0.0);
            for (i, &v) in x.iter().enumerate() {
                let ang = -TAU * (k * i) as f64 / n as f64;
                acc = acc + v * Complex::new(ang.cos(), ang.sin());
            }
            acc
        })
        .collect()
}

/// Amplitude of the DFT line at `bin` (so a tone of amplitude A on that bin
/// reads A).
fn line_amplitude(x: &[f64], bin: usize) -> f64 {
    let n = x.len() as f64;
    let (mut re, mut im) = (0.0, 0.0);
    for (i, &v) in x.iter().enumerate() {
        let ang = TAU * bin as f64 * i as f64 / n;
        re += v * ang.cos();
        im -= v * ang.sin();
    }
    2.0 * re.hypot(im) / n
}

/// Steady-state gain of an FIR/IIR filter at normalised frequency `f`
/// (1 = Nyquist), measured on a sine through the filter — independent of any
/// H(z) formula in the crate.
fn measured_gain(filter: impl Fn(&[f64]) -> Vec<f64>, f: f64) -> f64 {
    let n = 8192usize;
    // even bin so the second half still holds whole periods
    let bin = 2 * (f * n as f64 / 4.0).round() as usize;
    let x: Vec<f64> = (0..n)
        .map(|i| (TAU * bin as f64 * i as f64 / n as f64).sin())
        .collect();
    let y = filter(&x);
    // discard the transient, keep whole periods
    let tail = &y[n / 2..n];
    line_amplitude(tail, bin / 2)
}

/// Deterministic Gaussian-ish noise (sum of 12 uniforms, LCG), unit variance.
fn noise(seed: &mut u64) -> f64 {
    let mut s = 0.0;
    for _ in 0..12 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        s += (*seed >> 11) as f64 / (1u64 << 53) as f64;
    }
    s - 6.0
}

// ───────────────────────── FFT ────────────────────────────────────────────

#[test]
fn fft_matches_the_dft_definition_and_parseval() {
    let n = 256usize;
    let mut seed = 7u64;
    let x: Vec<Complex> = (0..n)
        .map(|_| Complex::new(noise(&mut seed), noise(&mut seed)))
        .collect();
    let reference = naive_dft(&x);
    let mut buf = x.clone();
    fft(&mut buf);
    for k in 0..n {
        let d = (buf[k] - reference[k]).mag();
        assert!(d < 1e-9 * n as f64, "bin {k}: fft vs DFT differ by {d}");
    }
    // Parseval
    let e_time: f64 = x.iter().map(|c| c.mag_sq()).sum();
    let e_freq: f64 = buf.iter().map(|c| c.mag_sq()).sum::<f64>() / n as f64;
    assert!(
        (e_time - e_freq).abs() < 1e-9 * e_time,
        "Parseval {e_time} vs {e_freq}"
    );
    // ifft ∘ fft = id
    ifft(&mut buf);
    for k in 0..n {
        assert!((buf[k] - x[k]).mag() < 1e-12, "round trip {k}");
    }
    // a real tone of amplitude A on bin k: |X_k| = |X_{N−k}| = A·N/2, rest 0
    let (a, k) = (0.75, 37usize);
    let mut tone: Vec<Complex> = (0..n)
        .map(|i| Complex::new(a * (TAU * k as f64 * i as f64 / n as f64).cos(), 0.0))
        .collect();
    fft(&mut tone);
    for (j, c) in tone.iter().enumerate() {
        let expected = if j == k || j == n - k {
            a * n as f64 / 2.0
        } else {
            0.0
        };
        assert!(
            (c.mag() - expected).abs() < 1e-9,
            "tone bin {j}: {} vs {expected}",
            c.mag()
        );
    }
    // linearity: fft(x + 2y) = fft(x) + 2 fft(y)
    let y: Vec<Complex> = (0..n)
        .map(|_| Complex::new(noise(&mut seed), 0.0))
        .collect();
    let mut fx = x.clone();
    fft(&mut fx);
    let mut fy = y.clone();
    fft(&mut fy);
    let mut fxy: Vec<Complex> = x.iter().zip(&y).map(|(&p, &q)| p + q * 2.0).collect();
    fft(&mut fxy);
    for k in 0..n {
        assert!(
            (fxy[k] - (fx[k] + fy[k] * 2.0)).mag() < 1e-9,
            "linearity {k}"
        );
    }
}

// ───────────────────────── windows ────────────────────────────────────────

#[test]
fn window_sums_endpoints_and_centres_match_their_closed_forms() {
    for n in [16usize, 63, 64, 1024] {
        let m = (n - 1) as f64;
        let (h, hm, b) = (hanning(n), hamming(n), blackman(n));
        // Σ over a symmetric window = DC term × n − (last cosine sample)
        let sum = |w: &[f64]| w.iter().sum::<f64>();
        assert!((sum(&h) - m / 2.0).abs() < 1e-9, "Hann sum n={n}");
        assert!(
            (sum(&hm) - (0.54 * n as f64 - 0.46)).abs() < 1e-9,
            "Hamming sum n={n}"
        );
        assert!((sum(&b) - 0.42 * m).abs() < 1e-9, "Blackman sum n={n}");
        // endpoints
        assert!(
            h[0].abs() < 1e-12 && h[n - 1].abs() < 1e-12,
            "Hann endpoints"
        );
        assert!(
            (hm[0] - 0.08).abs() < 1e-12 && (hm[n - 1] - 0.08).abs() < 1e-12,
            "Hamming endpoints"
        );
        assert!(
            b[0].abs() < 1e-12 && b[n - 1].abs() < 1e-12,
            "Blackman endpoints"
        );
        // symmetry and range
        for (w, name) in [(&h, "Hann"), (&hm, "Hamming"), (&b, "Blackman")] {
            for i in 0..n {
                assert!((w[i] - w[n - 1 - i]).abs() < 1e-12, "{name} symmetry {i}");
                assert!((-1e-12..=1.0 + 1e-12).contains(&w[i]), "{name} range {i}");
            }
        }
        if n % 2 == 1 {
            let c = n / 2;
            assert!(
                (h[c] - 1.0).abs() < 1e-12
                    && (hm[c] - 1.0).abs() < 1e-12
                    && (b[c] - 1.0).abs() < 1e-12,
                "centres"
            );
        }
    }
    // coherent gain of the Hann window on a tone: the windowed spectrum line
    // is (Σw/N)·A
    let n = 4096usize;
    let w = hanning(n);
    let x: Vec<f64> = (0..n)
        .map(|i| w[i] * (TAU * 100.0 * i as f64 / n as f64).sin())
        .collect();
    let gain = w.iter().sum::<f64>() / n as f64;
    assert!(
        (line_amplitude(&x, 100) - gain).abs() < 1e-6,
        "Hann coherent gain"
    );
    assert_eq!(hanning(1), vec![1.0]);
    assert!(hanning(0).is_empty());
}

// ───────────────────────── FIR ────────────────────────────────────────────

#[test]
fn windowed_sinc_fir_has_unit_dc_half_gain_at_cutoff_and_hamming_stopband() {
    let (order, fc) = (64usize, 0.3);
    let lp = fir_lowpass(order, fc);
    assert_eq!(lp.len(), order + 1);
    assert!((lp.iter().sum::<f64>() - 1.0).abs() < 1e-12, "DC gain 1");
    for i in 0..=order {
        assert!(
            (lp[i] - lp[order - i]).abs() < 1e-12,
            "linear phase symmetry {i}"
        );
    }
    let g = |f: f64| measured_gain(|x| fir_filter(x, &lp), f);
    assert!((g(0.05) - 1.0).abs() < 0.01, "passband {}", g(0.05));
    let at_fc = 20.0 * g(fc).log10();
    assert!(
        (at_fc + 6.0).abs() < 0.5,
        "−6 dB at cutoff, got {at_fc:.2} dB"
    );
    // Hamming transition ≈ 3.3/(order+1) of fs = 0.1 of Nyquist; beyond it ≤ −50 dB
    for f in [0.45, 0.6, 0.8, 0.95] {
        let db = 20.0 * g(f).log10();
        assert!(db < -50.0, "stopband {f}: {db:.1} dB");
    }
    // spectral inversion: DC 0, Nyquist 1
    let hp = fir_highpass(order, fc);
    assert!(hp.iter().sum::<f64>().abs() < 1e-12, "high-pass DC");
    let nyq: f64 = hp
        .iter()
        .enumerate()
        .map(|(i, &c)| if i % 2 == 0 { c } else { -c })
        .sum();
    // 1 − (low-pass leakage at Nyquist, ≤ −50 dB)
    assert!(
        (nyq.abs() - 1.0).abs() < 3e-3,
        "high-pass Nyquist gain {nyq}"
    );
    let gh = |f: f64| measured_gain(|x| fir_filter(x, &hp), f);
    assert!(20.0 * gh(0.05).log10() < -50.0, "high-pass rejects 0.05");
    assert!((gh(0.8) - 1.0).abs() < 0.01, "high-pass passes 0.8");
    // fir_filter with a unit impulse returns the taps
    let mut imp = vec![0.0; 8];
    imp[0] = 1.0;
    assert_eq!(&fir_filter(&imp, &lp)[..lp.len()], &lp[..]);
}

// ───────────────────────── IIR ────────────────────────────────────────────

#[test]
fn rbj_biquads_are_butterworth_at_q_inverse_sqrt2() {
    let fc = 0.2;
    let q = 1.0 / SQRT_2;
    let lp = Biquad::lowpass(fc, q);
    let g = |b: Biquad, f: f64| measured_gain(|x| b.filter(x), f);
    assert!((g(lp, 0.01) - 1.0).abs() < 1e-3, "LP DC");
    assert!(
        (g(lp, fc) - 1.0 / SQRT_2).abs() < 2e-3,
        "LP −3 dB at fc: {}",
        g(lp, fc)
    );
    assert!(g(lp, 0.95) < 0.02, "LP near Nyquist");
    // exact digital Butterworth via the prewarped bilinear map
    // Ω = tan(πf/2)/tan(πfc/2): |H_lp| = 1/√(1 + Ω⁴)
    let omega = |f: f64| (PI * f / 2.0).tan() / (PI * fc / 2.0).tan();
    for f in [fc / 2.0, 0.35, 0.6] {
        let expected = 1.0 / (1.0 + omega(f).powi(4)).sqrt();
        assert!(
            (g(lp, f) - expected).abs() < 2e-3,
            "LP {f}: {} vs {expected}",
            g(lp, f)
        );
    }
    let hp = Biquad::highpass(fc, q);
    assert!(g(hp, 0.01) < 0.01, "HP DC");
    assert!((g(hp, fc) - 1.0 / SQRT_2).abs() < 2e-3, "HP −3 dB at fc");
    assert!((g(hp, 0.9) - 1.0).abs() < 1e-2, "HP passes 0.9");
    let bp = Biquad::bandpass(fc, 2.0);
    assert!(
        (g(bp, fc) - 1.0).abs() < 2e-3,
        "BP unit peak at fc: {}",
        g(bp, fc)
    );
    assert!(g(bp, 0.01) < 0.05 && g(bp, 0.9) < 0.05, "BP skirts");
    // band edges: |Ω − 1/Ω| = 1/Q ⇒ Ω = (±1/Q + √(1/Q² + 4))/2, f = (2/π)·atan(Ω·tan(πfc/2))
    let q_bp = 2.0f64;
    for sign in [-1.0f64, 1.0] {
        let big_omega = (sign / q_bp + (1.0 / (q_bp * q_bp) + 4.0).sqrt()) / 2.0;
        let edge = 2.0 / PI * (big_omega * (PI * fc / 2.0).tan()).atan();
        let e = g(bp, edge);
        assert!((e - 1.0 / SQRT_2).abs() < 3e-3, "BP edge {edge}: {e}");
    }
    // the difference equation is exact: y[n] − b0 x[n] for an impulse is b1, b2
    let y = lp.filter(&[1.0, 0.0, 0.0]);
    assert!((y[0] - lp.b0).abs() < 1e-15);
    assert!((y[1] - (lp.b1 - lp.a1 * lp.b0)).abs() < 1e-15);
}

// ───────────────────────── PSD ────────────────────────────────────────────

#[test]
fn one_sided_periodogram_carries_the_full_signal_power() {
    let n = 1024usize;
    let mut seed = 11u64;
    let x: Vec<f64> = (0..n).map(|_| noise(&mut seed)).collect();
    let p = psd(&x);
    assert_eq!(p.len(), n / 2 + 1);
    let total: f64 = p.iter().sum();
    assert!(
        (total - energy(&x)).abs() < 1e-9 * energy(&x),
        "Σ psd = Σx²: {total} vs {}",
        energy(&x)
    );
    // a tone of amplitude A on bin k: psd[k] = A²N/2, all other bins 0
    let (a, k) = (0.5, 40usize);
    let tone: Vec<f64> = (0..n)
        .map(|i| a * (TAU * k as f64 * i as f64 / n as f64).sin())
        .collect();
    let p = psd(&tone);
    for (j, &v) in p.iter().enumerate() {
        let expected = if j == k { a * a * n as f64 / 2.0 } else { 0.0 };
        assert!(
            (v - expected).abs() < 1e-9 * a * a * n as f64,
            "tone psd[{j}] = {v} vs {expected}"
        );
    }
    // DC and Nyquist are single lines: x = 1 ⇒ psd[0] = N, x = (−1)^i ⇒ psd[N/2] = N
    assert!((psd(&vec![1.0; n])[0] - n as f64).abs() < 1e-9);
    let alt: Vec<f64> = (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    assert!((psd(&alt)[n / 2] - n as f64).abs() < 1e-9);
    // windowed: Parseval over the windowed samples
    let w = hamming(n);
    let pw = psd_windowed(&x, &w);
    let xw: Vec<f64> = x.iter().zip(&w).map(|(a, b)| a * b).collect();
    assert!(
        (pw.iter().sum::<f64>() - energy(&xw)).abs() < 1e-9 * energy(&xw),
        "windowed Parseval"
    );
}

// ───────────────────────── decimation ─────────────────────────────────────

#[test]
fn rate_changes_move_spectral_lines_by_the_rate_ratio() {
    let n = 1024usize;
    let k = 20usize;
    let x: Vec<f64> = (0..n)
        .map(|i| (TAU * k as f64 * i as f64 / n as f64).cos())
        .collect();
    // decimate by 4: N/4 samples, tone stays on bin k (below the new Nyquist)
    let d = decimate(&x, 4);
    assert_eq!(d.len(), n / 4);
    assert!((line_amplitude(&d, k) - 1.0).abs() < 1e-9, "decimated tone");
    assert_eq!(decimate(&x, 1), x);
    // zero-insertion by 3: 3N samples, tone on bin k plus images at N ± k, 2N ± k,
    // each with amplitude 1/3
    let u = interpolate(&x, 3);
    assert_eq!(u.len(), 3 * n);
    for bin in [k, n - k, n + k, 2 * n - k, 2 * n + k] {
        assert!(
            (line_amplitude(&u, bin) - 1.0 / 3.0).abs() < 1e-9,
            "image at {bin}"
        );
    }
    assert!(line_amplitude(&u, k + 1) < 1e-9);
    // linear interpolation reproduces a linear function exactly
    let ramp: Vec<f64> = (0..10).map(|i| 3.0 * i as f64 - 2.0).collect();
    let li = interpolate_linear(&ramp, 4);
    assert_eq!(li.len(), 9 * 4 + 1);
    for (j, &v) in li.iter().enumerate() {
        let expected = 3.0 * j as f64 / 4.0 - 2.0;
        assert!(
            (v - expected).abs() < 1e-12,
            "linear interp {j}: {v} vs {expected}"
        );
    }
    assert_eq!(interpolate_linear(&[5.0], 4), vec![5.0]);
}

// ───────────────────────── convolution ────────────────────────────────────

#[test]
fn convolution_and_correlation_identities() {
    let a = [1.0, -2.0, 3.5, 0.25, 4.0];
    let delta = [1.0];
    assert_eq!(convolve(&a, &delta), a.to_vec());
    let shifted = convolve(&a, &[0.0, 0.0, 1.0]);
    assert_eq!(&shifted[2..], &a[..]);
    // box ⊛ box = triangle 1,2,3,4,3,2,1
    let t = convolve(&[1.0; 4], &[1.0; 4]);
    assert_eq!(t, vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]);
    // commutative, Σ(a⊛b) = Σa·Σb
    let b = [0.5, 2.0, -1.0];
    assert_eq!(convolve(&a, &b), convolve(&b, &a));
    let s: f64 = convolve(&a, &b).iter().sum();
    assert!((s - a.iter().sum::<f64>() * b.iter().sum::<f64>()).abs() < 1e-12);
    // correlate(a, b) = convolve(a, reverse b); zero lag sits at index len(b)−1
    let rb: Vec<f64> = b.iter().rev().copied().collect();
    assert_eq!(correlate(&a, &b), convolve(&a, &rb));
    let ac = correlate(&a, &a);
    assert!(
        (ac[a.len() - 1] - energy(&a)).abs() < 1e-12,
        "zero-lag autocorrelation = energy"
    );
    for lag in 0..a.len() {
        assert!(
            (ac[a.len() - 1 + lag] - ac[a.len() - 1 - lag]).abs() < 1e-12,
            "autocorrelation even"
        );
        assert!(
            ac[a.len() - 1 + lag] <= ac[a.len() - 1] + 1e-12,
            "peak at zero lag"
        );
    }
    assert!(convolve(&a, &[]).is_empty() && correlate(&[], &a).is_empty());
}

// ───────────────────────── wavelets ───────────────────────────────────────

#[test]
fn haar_and_db4_are_orthonormal_with_perfect_reconstruction() {
    let n = 64usize;
    let mut seed = 3u64;
    let x: Vec<f64> = (0..n).map(|_| noise(&mut seed)).collect();
    let e = energy(&x);
    for (fwd, inv, name) in [
        (
            haar_forward as fn(&mut [f64]),
            haar_inverse as fn(&mut [f64]),
            "haar",
        ),
        (haar_forward_multi, haar_inverse_multi, "haar multi"),
        (db4_forward, db4_inverse, "db4"),
    ] {
        let mut d = x.clone();
        fwd(&mut d);
        assert!(
            (energy(&d) - e).abs() < 1e-9 * e,
            "{name} Parseval {} vs {e}",
            energy(&d)
        );
        inv(&mut d);
        for i in 0..n {
            assert!((d[i] - x[i]).abs() < 1e-9, "{name} reconstruction {i}");
        }
    }
    // a constant c has zero detail and approximation c·√2 (Σh = √2)
    for (fwd, name) in [
        (haar_forward as fn(&mut [f64]), "haar"),
        (db4_forward, "db4"),
    ] {
        let mut c = vec![2.5; n];
        fwd(&mut c);
        for i in 0..n / 2 {
            assert!(
                (c[i] - 2.5 * SQRT_2).abs() < 1e-12,
                "{name} approx {i}: {}",
                c[i]
            );
            assert!(
                c[n / 2 + i].abs() < 1e-12,
                "{name} detail {i}: {}",
                c[n / 2 + i]
            );
        }
    }
    // Haar explicit: [a, b] → [(a+b)/√2, (a−b)/√2]
    let mut p = [3.0, 1.0];
    haar_forward(&mut p);
    assert!((p[0] - 4.0 / SQRT_2).abs() < 1e-12 && (p[1] - 2.0 / SQRT_2).abs() < 1e-12);
    // full multi-level Haar of a constant leaves only the DC coefficient c·√N
    let mut c = vec![1.0; n];
    haar_forward_multi(&mut c);
    assert!((c[0] - (n as f64).sqrt()).abs() < 1e-9);
    assert!(c[1..].iter().all(|v| v.abs() < 1e-12));
    // a linear ramp has zero db4 detail away from the periodic wrap (2 vanishing moments)
    let ramp: Vec<f64> = (0..n).map(|i| 0.5 * i as f64 + 1.0).collect();
    let mut r = ramp.clone();
    db4_forward(&mut r);
    for i in 0..n / 2 - 2 {
        assert!(
            r[n / 2 + i].abs() < 1e-9,
            "db4 ramp detail {i}: {}",
            r[n / 2 + i]
        );
    }
}

// ───────────────────────── Kalman ─────────────────────────────────────────

#[test]
fn scalar_kalman_reaches_the_riccati_fixed_point_and_the_sample_mean() {
    let (q, r) = (0.01, 1.0);
    let mut kf = KalmanFilter::new(
        vec![0.0],
        vec![100.0],
        vec![1.0],
        vec![q],
        vec![1.0],
        vec![r],
    )
    .unwrap();
    let p_prior = (q + (q * q + 4.0 * q * r).sqrt()) / 2.0;
    let p_post = p_prior * r / (p_prior + r);
    for _ in 0..2000 {
        kf.predict();
        kf.update(&[1.0]).unwrap();
    }
    assert!(
        (kf.covariance()[0] - p_post).abs() < 1e-9,
        "P∞ {} vs {p_post}",
        kf.covariance()[0]
    );
    // with q = 0 the filter is the running mean: P_n = P0 r/(r + n P0), x = mean(z)
    let (p0, r) = (4.0, 2.0);
    let mut kf = KalmanFilter::new(
        vec![0.0],
        vec![p0],
        vec![1.0],
        vec![0.0],
        vec![1.0],
        vec![r],
    )
    .unwrap();
    let mut seed = 5u64;
    let mut sum = 0.0;
    for n in 1..=500u32 {
        let z = 3.0 + noise(&mut seed);
        sum += z;
        kf.predict();
        kf.update(&[z]).unwrap();
        let p_n = p0 * r / (r + n as f64 * p0);
        assert!((kf.covariance()[0] - p_n).abs() < 1e-12, "P_{n}");
    }
    // x_n = (P0/r · Σz) / (1 + n P0/r): with P0 ≫ r/n it is the sample mean
    let n = 500.0;
    let expected = (p0 / r * sum) / (1.0 + n * p0 / r);
    assert!(
        (kf.state()[0] - expected).abs() < 1e-9,
        "x {} vs {expected}",
        kf.state()[0]
    );
    assert!((kf.state()[0] - sum / n).abs() < 0.01, "≈ sample mean");
    // innovation is z − H x
    let inn = kf.innovation(&[10.0]).unwrap();
    assert!((inn[0] - (10.0 - kf.state()[0])).abs() < 1e-12);
    assert!(KalmanFilter::new(
        vec![0.0],
        vec![1.0, 2.0],
        vec![1.0],
        vec![0.0],
        vec![1.0],
        vec![1.0]
    )
    .is_none());
}

// ───────────────────────── tracking ───────────────────────────────────────

#[test]
fn discriminators_and_loop_filter_follow_kaplan_hegarty() {
    for phi in [-1.4f64, -0.7, -0.1, 0.0, 0.3, 1.2] {
        let d = pll_costas_discriminator(phi.cos(), phi.sin());
        assert!((d - phi).abs() < 1e-12, "Costas {phi}");
        // data-bit flip (180°) gives the same output
        let f = pll_costas_discriminator(-phi.cos(), -phi.sin());
        assert!((f - phi).abs() < 1e-12, "Costas flip {phi}");
    }
    assert!((pll_costas_discriminator(0.0, 1.0) - PI / 2.0).abs() < 1e-12);
    // FLL: two prompts Δφ apart over dt read Δφ/(2π dt) Hz
    let (dt, f_hz) = (1e-3, 25.0);
    let dphi = TAU * f_hz * dt;
    let est = fll_cross_product_discriminator(1.0, 0.0, dphi.cos(), dphi.sin(), dt);
    assert!((est - f_hz).abs() < 1e-9, "FLL {est}");
    assert_eq!(
        fll_cross_product_discriminator(1.0, 0.0, 0.0, 1.0, 0.0),
        0.0
    );
    // loop coefficients: invert Table 5.4 — Bn = ω_n (4ξ² + 1)/(8ξ), ω_n = √k2, ξ = k1/(2ω_n)
    for (bn, xi) in [(15.0, 0.707), (2.0, 1.0), (25.0, 0.5)] {
        let l = SecondOrderLoop::new(bn, xi, dt);
        let omega_n = l.k2.sqrt();
        let xi_back = l.k1 / (2.0 * omega_n);
        let bn_back = omega_n * (4.0 * xi_back * xi_back + 1.0) / (8.0 * xi_back);
        assert!(
            (xi_back - xi).abs() < 1e-12 && (bn_back - bn).abs() < 1e-9,
            "Bn {bn} ξ {xi}"
        );
    }
    // step response to a constant error e: out_n = k1 e + k2 e n dt
    let mut l = SecondOrderLoop::new(15.0, 0.707, dt);
    let e = 0.2;
    for n in 1..=100u32 {
        let out = l.update(e);
        let expected = l.k1 * e + l.k2 * e * n as f64 * dt;
        assert!((out - expected).abs() < 1e-12, "step {n}");
    }
    l.reset();
    assert_eq!(l.update(0.0), 0.0);
}

// ───────────────────────── GNSS ───────────────────────────────────────────

#[test]
fn ca_codes_match_is_gps_200_and_the_gold_code_correlation_theorem() {
    // IS-GPS-200 Table 3-Ia: first 10 chips (octal)
    let first_10 = [
        (1u16, 0o1440u16),
        (2, 0o1620),
        (3, 0o1710),
        (4, 0o1744),
        (5, 0o1133),
        (6, 0o1455),
        (7, 0o1131),
        (8, 0o1454),
        (9, 0o1626),
        (10, 0o1504),
    ];
    for (prn, octal) in first_10 {
        let code = ca_code(prn).unwrap();
        assert_eq!(code.len(), CA_CODE_LENGTH);
        for i in 0..10 {
            let bit = (octal >> (9 - i)) & 1;
            let expected = if bit == 1 { -1 } else { 1 };
            assert_eq!(code[i], expected, "PRN {prn} chip {i}");
        }
        // balance: 512 ones, 511 zeros
        let ones = code.iter().filter(|&&c| c == -1).count();
        assert_eq!(ones, 512, "PRN {prn} balance");
        assert!(code.iter().all(|&c| c == 1 || c == -1));
    }
    // Gold code (n = 10) periodic correlation takes only {−65, −1, 63}/1023 off-peak
    let allowed = |v: f64| {
        [-65.0, -1.0, 63.0]
            .iter()
            .any(|t| (v * 1023.0 - t).abs() < 1e-9)
    };
    let c1 = ca_code(1).unwrap();
    let c2 = ca_code(2).unwrap();
    assert_eq!(normalised_correlation(&c1, &c1), Some(1.0));
    for shift in 1..CA_CODE_LENGTH {
        let rotated: Vec<i8> = c1
            .iter()
            .cycle()
            .skip(shift)
            .take(CA_CODE_LENGTH)
            .copied()
            .collect();
        let auto = normalised_correlation(&c1, &rotated).unwrap();
        assert!(
            allowed(auto),
            "PRN 1 autocorrelation at shift {shift}: {}",
            auto * 1023.0
        );
        let cross = normalised_correlation(&c2, &rotated).unwrap();
        assert!(
            allowed(cross),
            "PRN 1×2 cross-correlation at shift {shift}: {}",
            cross * 1023.0
        );
    }
    assert!(allowed(normalised_correlation(&c1, &c2).unwrap()));
    assert!(ca_code(33).is_none() && ca_code(0).is_none());
    assert_eq!(normalised_correlation(&[1, -1], &[1]), None);
}

#[test]
fn nwpr_cn0_estimate_recovers_the_synthesised_carrier_to_noise_ratio() {
    // I = A + n_I, Q = n_Q with unit-variance noise: C/N0 = 10 log10(A²/(2σ²)) − 10 log10(T)
    let cfg = Cn0Config {
        samples_per_window: 20,
        windows: 400,
        coherent_integration_s: 1e-3,
    };
    let mut seed = 17u64;
    for a in [3.0, 5.0, 10.0] {
        let n = cfg.samples_per_window * cfg.windows;
        let i: Vec<f64> = (0..n).map(|_| a + noise(&mut seed)).collect();
        let q: Vec<f64> = (0..n).map(|_| noise(&mut seed)).collect();
        let est = estimate_cn0(&i, &q, cfg).unwrap();
        let expected = 10.0 * (a * a / 2.0).log10() + 30.0;
        assert!(
            (est - expected).abs() < 0.5,
            "A={a}: {est:.2} vs {expected:.2} dB-Hz"
        );
    }
    // noise-free input saturates the estimator (NP = M) → None, not +∞
    let i = vec![1.0; 40];
    let q = vec![0.0; 40];
    assert!(estimate_cn0(
        &i,
        &q,
        Cn0Config {
            samples_per_window: 20,
            windows: 2,
            coherent_integration_s: 1e-3
        }
    )
    .is_none());
    assert!(estimate_cn0(&i, &q[..39], cfg).is_none());
}

// ───────────────────────── utility ────────────────────────────────────────

#[test]
fn energy_rms_and_padding_closed_forms() {
    let n = 1000usize;
    let a = 2.0;
    let x: Vec<f64> = (0..n)
        .map(|i| a * (TAU * 10.0 * i as f64 / n as f64).sin())
        .collect();
    assert!(
        (energy(&x) - a * a * n as f64 / 2.0).abs() < 1e-9,
        "sine energy A²N/2"
    );
    assert!((rms(&x) - a / SQRT_2).abs() < 1e-12, "sine rms A/√2");
    assert_eq!(rms(&[]), 0.0);
    let p = zero_pad_to_power_of_two(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(p, vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0]);
    assert_eq!(zero_pad_to_power_of_two(&[1.0; 8]).len(), 8);
    assert_eq!(zero_pad_to_power_of_two(&[]), vec![0.0]);
}
