//! Oracle: the FFT, the windows and the complex magnitude give the same bits on every target.
//!
//! The inputs are built from integer arithmetic only, and the outputs go through IEEE 754
//! operations and `alice_det_math` (no platform libm, no fused multiply-add from the compiler), so
//! the 64-bit FNV-1a of the output bits is a constant. CI runs this on Linux, macOS and Windows;
//! a different digest on any of them means a platform-dependent path came back.
//!
//! The digests were recorded on aarch64-apple-darwin.

use alice_signal::complex::Complex;
use alice_signal::fft::fft;
use alice_signal::windows::{blackman, hamming, hanning};

fn fnv1a(values: impl Iterator<Item = f64>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in values {
        for b in v.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// Values in [-0.5, 0.5) from integers only.
fn input(n: usize) -> Vec<Complex> {
    (0..n)
        .map(|j| {
            let a = ((j * 7919) % 1001) as f64 / 1000.0 - 0.5;
            let b = ((j * 104_729) % 997) as f64 / 996.0 - 0.5;
            Complex::new(a, b)
        })
        .collect()
}

fn digest_of(c: &[Complex]) -> u64 {
    fnv1a(c.iter().flat_map(|z| [z.re, z.im]))
}

#[test]
fn fft_bits_are_pinned() {
    let mut cases = 0;
    for (n, want) in [
        (1usize, 0x6e8a_b90a_94bb_5e45_u64),
        (64, 0xc09e_3e62_2f77_45bf),
        (4096, 0xd525_10b4_7618_0b25),
    ] {
        let mut x = input(n);
        fft(&mut x);
        assert_eq!(digest_of(&x), want, "fft n {n}: {:#018x}", digest_of(&x));
        cases += 1;
    }
    assert_eq!(cases, 3);
}

#[test]
fn window_and_magnitude_bits_are_pinned() {
    let got = [
        fnv1a(hanning(1024).into_iter()),
        fnv1a(hamming(1000).into_iter()),
        fnv1a(blackman(513).into_iter()),
        fnv1a(input(257).iter().map(|z| z.mag())),
        fnv1a((0..200).map(|k| Complex::from_polar(1.5, f64::from(k) * 0.0625).re)),
    ];
    let want = [
        0xbb8e_6e03_e8e6_7a3c_u64,
        0x5812_d140_8b08_c3a4,
        0x14b1_f515_2bbf_4937,
        0xc0a1_2b27_f4f8_a101,
        0x4283_4f38_0006_13bf,
    ];
    assert_eq!(got, want, "got {got:#018x?}");
}
