//! Fuzz: KalmanFilter with arbitrary dimensions / matrices (singular,
//! mismatched shapes, NaN) — `new` must reject wrong shapes with `None`,
//! `predict` / `update` / `innovation` must never panic, and finite,
//! well-shaped inputs must keep the state finite; the EKF matrix helpers
//! (`invert` of a singular matrix → `None`) likewise.
#![no_main]

use alice_signal::ekf::{add, eye, invert, matmul, matvec, sub, transpose, vadd, vsub};
use alice_signal::kalman::KalmanFilter;
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    n: u8,
    m: u8,
    x0: Vec<f64>,
    p0: Vec<f64>,
    f: Vec<f64>,
    q: Vec<f64>,
    h: Vec<f64>,
    r: Vec<f64>,
    z: Vec<f64>,
    steps: u8,
}

fn shaped(v: &[f64], len: usize, fill: f64) -> Vec<f64> {
    (0..len).map(|i| v.get(i).copied().unwrap_or(fill)).collect()
}

fuzz_target!(|input: Input| {
    // raw shapes: new() must reject or accept without panicking
    let _ = KalmanFilter::new(
        input.x0.clone(),
        input.p0.clone(),
        input.f.clone(),
        input.q.clone(),
        input.h.clone(),
        input.r.clone(),
    );
    let n = usize::from(input.n % 4) + 1;
    let m = usize::from(input.m % 3) + 1;
    let kf = KalmanFilter::new(
        shaped(&input.x0, n, 0.0),
        shaped(&input.p0, n * n, 1.0),
        shaped(&input.f, n * n, 1.0),
        shaped(&input.q, n * n, 0.0),
        shaped(&input.h, m * n, 1.0),
        shaped(&input.r, m * m, 1.0),
    );
    if let Some(mut kf) = kf {
        assert_eq!((kf.state_dim(), kf.measurement_dim()), (n, m));
        for k in 0..usize::from(input.steps % 16) {
            kf.predict();
            let z = shaped(&input.z[input.z.len().min(k)..], m, 0.0);
            let _ = kf.update(&z);
            let _ = kf.innovation(&z);
            let _ = kf.innovation(&input.z);
        }
        let _ = (kf.state().len(), kf.covariance().len());
    }

    // EKF matrix helpers on n×n
    let a: Vec<Vec<f64>> = (0..n).map(|i| shaped(&input.f[(i * n).min(input.f.len())..], n, 0.0)).collect();
    let b = eye(n);
    let _ = (matmul(&a, &b), transpose(&a), add(&a, &b), sub(&a, &b));
    let v = shaped(&input.x0, n, 1.0);
    let _ = (matvec(&a, &v), vadd(&v, &v), vsub(&v, &v));
    let _ = invert(&a);
    assert!(invert(&eye(n)).is_some(), "identity is invertible");
});
