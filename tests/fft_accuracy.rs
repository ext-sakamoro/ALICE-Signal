//! Oracle: the FFT's error against the DFT definition grows like `ε · log₂ n · ‖x‖`.
//!
//! A radix-2 FFT whose twiddle factors are each correctly rounded has an error bound of order
//! `ε · log₂ n · ‖x‖₂` per output (Higham, "Accuracy and Stability of Numerical Algorithms",
//! §24.1). Twiddles produced by a recurrence (`w ← w · w_n`) accumulate rounding along each stage
//! and the error grows roughly like `ε · n · ‖x‖`, which this test rejects at large `n`.
//!
//! The reference is the DFT definition evaluated in f64 with the angle reduced exactly
//! (`(j · k) mod n` before scaling) and compensated (Neumaier) summation, so its own error is a few
//! ulp of `‖x‖`, well below the bound. It uses the platform `sin` / `cos` on purpose: it must not
//! share the implementation's twiddle source.
//!
//! Bound: every sampled bin within `4 · ε · log₂ n · ‖x‖₂`.

// the reference uses the platform libm on purpose (an independent twiddle source)
#![allow(clippy::disallowed_methods)]

use alice_signal::complex::Complex;
use alice_signal::fft::fft;
use std::f64::consts::TAU;

/// Uniform values in [-0.5, 0.5) from a 64-bit LCG.
fn signal(n: usize, seed: u64) -> Vec<Complex> {
    let mut s = seed;
    let mut next = || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    (0..n).map(|_| Complex::new(next(), next())).collect()
}

/// Neumaier-compensated sum.
fn sum(values: impl Iterator<Item = f64>) -> f64 {
    let (mut s, mut c) = (0.0_f64, 0.0_f64);
    for v in values {
        let t = s + v;
        c += if s.abs() >= v.abs() {
            (s - t) + v
        } else {
            (v - t) + s
        };
        s = t;
    }
    s + c
}

fn dft_bin(x: &[Complex], k: usize) -> Complex {
    let n = x.len();
    let terms = || {
        x.iter().enumerate().map(move |(j, c)| {
            let a = -TAU * ((j * k) % n) as f64 / n as f64;
            (c, a.cos(), a.sin())
        })
    };
    Complex::new(
        sum(terms().flat_map(|(c, co, si)| [c.re * co, -c.im * si])),
        sum(terms().flat_map(|(c, co, si)| [c.re * si, c.im * co])),
    )
}

#[test]
fn error_grows_like_eps_log_n() {
    let mut checked = 0;
    for (n, seed) in [(256usize, 1u64), (1024, 2), (4096, 3), (16384, 4)] {
        let x = signal(n, seed);
        let norm = x.iter().map(|c| c.mag_sq()).sum::<f64>().sqrt();
        let bound = 4.0 * f64::EPSILON * (n as f64).log2() * norm;
        let mut y = x.clone();
        fft(&mut y);
        for k in (0..n).step_by(n / 64).chain([1, n / 2 - 1, n / 2, n - 1]) {
            let d = (y[k] - dft_bin(&x, k)).mag();
            assert!(
                d <= bound,
                "n {n} bin {k}: error {d:e} > bound {bound:e} ({:.1} × ε·log₂n·‖x‖)",
                d / (f64::EPSILON * (n as f64).log2() * norm)
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 4 * (64 + 4));
}
