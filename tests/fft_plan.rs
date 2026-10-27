//! Oracle: an `FftPlan` built once and reused gives exactly the bits of `fft` / `ifft`.
//!
//! The plan only moves the twiddle table out of the call, so every output must be bit-identical
//! to the one-shot functions (which `tests/bits_pinned.rs` pins), for every size and across many
//! buffers transformed with the same plan. A buffer whose length is not the plan's is refused.

use alice_signal::complex::Complex;
use alice_signal::fft::{fft, ifft, FftPlan};

/// Values in [-0.5, 0.5) from integers only.
fn input(n: usize, salt: usize) -> Vec<Complex> {
    (0..n)
        .map(|j| {
            let a = (((j + salt) * 7919) % 1001) as f64 / 1000.0 - 0.5;
            let b = (((j + 3 * salt) * 104_729) % 997) as f64 / 996.0 - 0.5;
            Complex::new(a, b)
        })
        .collect()
}

fn bits(c: &[Complex]) -> Vec<(u64, u64)> {
    c.iter().map(|z| (z.re.to_bits(), z.im.to_bits())).collect()
}

#[test]
fn a_reused_plan_gives_the_bits_of_fft_and_ifft() {
    let mut compared = 0;
    for log_n in 0..=12 {
        let n = 1usize << log_n;
        let plan = FftPlan::new(n);
        assert_eq!(plan.len(), n);
        for salt in 0..3 {
            let x = input(n, salt);
            let mut once = x.clone();
            fft(&mut once);
            let mut planned = x.clone();
            plan.forward(&mut planned);
            assert_eq!(bits(&planned), bits(&once), "forward n {n} salt {salt}");

            let mut back_once = once.clone();
            ifft(&mut back_once);
            plan.inverse(&mut planned);
            assert_eq!(bits(&planned), bits(&back_once), "inverse n {n} salt {salt}");
            compared += 1;
        }
    }
    assert_eq!(compared, 13 * 3);
}

#[test]
#[should_panic(expected = "length")]
fn a_buffer_of_another_length_is_refused() {
    let plan = FftPlan::new(8);
    let mut buf = input(16, 0);
    plan.forward(&mut buf);
}

#[test]
#[should_panic(expected = "power of two")]
fn a_plan_needs_a_power_of_two() {
    let _ = FftPlan::new(12);
}

#[test]
#[should_panic(expected = "power of two")]
fn a_plan_needs_a_nonzero_length() {
    let _ = FftPlan::new(0);
}
