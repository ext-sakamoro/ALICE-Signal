//! Extended Kalman Filter (EKF).
//!
//! The EKF is the workhorse non-linear estimator in GNSS positioning
//! solutions: the observation model (pseudorange vs. receiver clock and
//! satellite position) is non-linear, so a linearised Jacobian is
//! computed at every measurement update.
//!
//! Standard EKF cycle:
//!
//! ```text
//! Predict:  x̂⁻ = f(x̂, Δt),  P⁻ = F·P·Fᵀ + Q
//! Update:   K  = P⁻·Hᵀ·(H·P⁻·Hᵀ + R)⁻¹
//!           x̂  = x̂⁻ + K·(z − h(x̂⁻))
//!           P  = (I − K·H)·P⁻
//! ```
//!
//! This implementation uses `Vec<Vec<f64>>` matrices — clear, dependency
//! free, and adequate for typical GNSS state dimensions (N ≤ 15).
//!
//! # References
//!
//! - Kalman, R. E. (1960), "A new approach to linear filtering and
//!   prediction problems", ASME J. Basic Eng., 82(D), 35-45.
//! - Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001), "Estimation
//!   with Applications to Tracking and Navigation", Wiley, §5.
//! - Kaplan, E. D. & Hegarty, C. J. (2017), §11 Integration of GPS
//!   with other sensors.

#![allow(
    clippy::doc_markdown,
    clippy::missing_panics_doc,
    clippy::needless_pass_by_value
)]

// ---------------------------------------------------------------------------
// Matrix
// ---------------------------------------------------------------------------

/// A row-major dense f64 matrix.
pub type Matrix = Vec<Vec<f64>>;
/// A dense f64 vector.
pub type Vector = Vec<f64>;

/// Construct an `n × n` identity matrix.
#[must_use]
pub fn eye(n: usize) -> Matrix {
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n {
        m[i][i] = 1.0;
    }
    m
}

/// Matrix multiplication `A (m×k) · B (k×n) = C (m×n)`.
///
/// Returns an empty matrix if the shapes disagree.
#[must_use]
pub fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
    if a.is_empty() || b.is_empty() {
        return Matrix::new();
    }
    let m = a.len();
    let k = a[0].len();
    let n = b[0].len();
    if b.len() != k {
        return Matrix::new();
    }
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for p in 0..k {
                s += a[i][p] * b[p][j];
            }
            c[i][j] = s;
        }
    }
    c
}

/// Matrix transpose.
#[must_use]
pub fn transpose(a: &Matrix) -> Matrix {
    if a.is_empty() {
        return Matrix::new();
    }
    let m = a.len();
    let n = a[0].len();
    let mut b = vec![vec![0.0; m]; n];
    for i in 0..m {
        for j in 0..n {
            b[j][i] = a[i][j];
        }
    }
    b
}

/// Matrix addition.
#[must_use]
pub fn add(a: &Matrix, b: &Matrix) -> Matrix {
    if a.is_empty() {
        return b.clone();
    }
    if b.is_empty() {
        return a.clone();
    }
    let m = a.len();
    let n = a[0].len();
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    c
}

/// Matrix subtraction.
#[must_use]
pub fn sub(a: &Matrix, b: &Matrix) -> Matrix {
    let m = a.len();
    let n = a[0].len();
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            c[i][j] = a[i][j] - b[i][j];
        }
    }
    c
}

/// Matrix · vector.
#[must_use]
pub fn matvec(a: &Matrix, v: &Vector) -> Vector {
    let m = a.len();
    let n = a[0].len();
    let mut out = vec![0.0; m];
    for i in 0..m {
        let mut s = 0.0;
        for j in 0..n {
            s += a[i][j] * v[j];
        }
        out[i] = s;
    }
    out
}

/// Vector addition `a + b` (element-wise).
#[must_use]
pub fn vadd(a: &Vector, b: &Vector) -> Vector {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Vector subtraction `a - b` (element-wise).
#[must_use]
pub fn vsub(a: &Vector, b: &Vector) -> Vector {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Invert a square matrix via Gauss-Jordan with partial pivoting.
/// Returns `None` on singular input.
#[must_use]
pub fn invert(a: &Matrix) -> Option<Matrix> {
    let n = a.len();
    if n == 0 || a[0].len() != n {
        return None;
    }
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n + i] = 1.0;
    }
    for i in 0..n {
        let mut max_row = i;
        let mut max_val = aug[i][i].abs();
        for k in (i + 1)..n {
            if aug[k][i].abs() > max_val {
                max_val = aug[k][i].abs();
                max_row = k;
            }
        }
        if max_val < 1e-12 {
            return None;
        }
        aug.swap(i, max_row);
        let pivot = aug[i][i];
        for j in 0..(2 * n) {
            aug[i][j] /= pivot;
        }
        for k in 0..n {
            if k == i {
                continue;
            }
            let factor = aug[k][i];
            for j in 0..(2 * n) {
                aug[k][j] -= factor * aug[i][j];
            }
        }
    }
    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[i][n + j];
        }
    }
    Some(inv)
}

// ---------------------------------------------------------------------------
// EKF Model
// ---------------------------------------------------------------------------

/// EKF system model.
pub trait EkfModel {
    /// State dimension.
    fn state_dim(&self) -> usize;
    /// Measurement dimension.
    fn meas_dim(&self) -> usize;
    /// Non-linear state transition `f(x, Δt)`.
    fn f(&self, x: &Vector, dt_s: f64) -> Vector;
    /// Jacobian `F = ∂f/∂x`, size `N × N`.
    fn jacobian_f(&self, x: &Vector, dt_s: f64) -> Matrix;
    /// Non-linear measurement `h(x)`.
    fn h(&self, x: &Vector) -> Vector;
    /// Jacobian `H = ∂h/∂x`, size `M × N`.
    fn jacobian_h(&self, x: &Vector) -> Matrix;
}

// ---------------------------------------------------------------------------
// StateEstimate
// ---------------------------------------------------------------------------

/// EKF state estimate.
#[derive(Debug, Clone, PartialEq)]
pub struct StateEstimate {
    /// State mean `(N)`.
    pub x: Vector,
    /// State covariance `(N × N)`.
    pub p: Matrix,
}

impl StateEstimate {
    /// Construct a new estimate.
    #[must_use]
    pub fn new(x: Vector, p: Matrix) -> Self {
        Self { x, p }
    }

    /// Trace of the covariance matrix.
    #[must_use]
    pub fn trace(&self) -> f64 {
        (0..self.p.len()).map(|i| self.p[i][i]).sum()
    }
}

// ---------------------------------------------------------------------------
// Predict / Update
// ---------------------------------------------------------------------------

/// EKF prediction step.
#[must_use]
pub fn predict<M: EkfModel>(
    model: &M,
    state: StateEstimate,
    dt_s: f64,
    q: &Matrix,
) -> StateEstimate {
    let x_pred = model.f(&state.x, dt_s);
    let f = model.jacobian_f(&state.x, dt_s);
    let f_p = matmul(&f, &state.p);
    let f_p_ft = matmul(&f_p, &transpose(&f));
    let p_pred = add(&f_p_ft, q);
    StateEstimate {
        x: x_pred,
        p: p_pred,
    }
}

/// EKF update step. If innovation covariance `S` is singular, returns
/// the input state unchanged.
#[must_use]
pub fn update<M: EkfModel>(
    model: &M,
    state: StateEstimate,
    z: &Vector,
    r: &Matrix,
) -> StateEstimate {
    let h = model.jacobian_h(&state.x); // M × N
    let ht = transpose(&h); // N × M
    let z_pred = model.h(&state.x); // M
    let hp = matmul(&h, &state.p); // M × N
    let hp_ht = matmul(&hp, &ht); // M × M
    let s = add(&hp_ht, r); // M × M
    let Some(s_inv) = invert(&s) else {
        return state;
    };
    let p_ht = matmul(&state.p, &ht); // N × M
    let k = matmul(&p_ht, &s_inv); // N × M
    let y = vsub(z, &z_pred); // M
    let dx = matvec(&k, &y); // N
    let x_new = vadd(&state.x, &dx);
    let n = state.x.len();
    let kh = matmul(&k, &h); // N × N
    let i_kh = sub(&eye(n), &kh);
    let p_new = matmul(&i_kh, &state.p);
    StateEstimate { x: x_new, p: p_new }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct ConstVel;

    impl EkfModel for ConstVel {
        fn state_dim(&self) -> usize {
            2
        }
        fn meas_dim(&self) -> usize {
            1
        }
        fn f(&self, x: &Vector, dt_s: f64) -> Vector {
            vec![x[1].mul_add(dt_s, x[0]), x[1]]
        }
        fn jacobian_f(&self, _x: &Vector, dt_s: f64) -> Matrix {
            vec![vec![1.0, dt_s], vec![0.0, 1.0]]
        }
        fn h(&self, x: &Vector) -> Vector {
            vec![x[0]]
        }
        fn jacobian_h(&self, _x: &Vector) -> Matrix {
            vec![vec![1.0, 0.0]]
        }
    }

    #[test]
    fn predict_advances_position_by_velocity() {
        let s = StateEstimate::new(vec![0.0, 1.0], eye(2));
        let s2 = predict(&ConstVel, s, 1.0, &vec![vec![0.0; 2]; 2]);
        assert!((s2.x[0] - 1.0).abs() < 1e-12);
        assert!((s2.x[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn predict_grows_covariance_by_q() {
        let s = StateEstimate::new(vec![0.0, 0.0], eye(2));
        let q = vec![vec![0.1, 0.0], vec![0.0, 0.1]];
        let s2 = predict(&ConstVel, s, 1.0, &q);
        assert!(s2.p[0][0] > 1.0);
        assert!(s2.p[1][1] > 1.0);
    }

    #[test]
    fn update_reduces_covariance() {
        let s = StateEstimate::new(vec![0.0, 0.0], vec![vec![10.0, 0.0], vec![0.0, 10.0]]);
        let before = s.trace();
        let s2 = update(&ConstVel, s, &vec![5.0], &vec![vec![0.1]]);
        assert!(s2.trace() < before);
    }

    #[test]
    fn tracking_loop_converges_on_ground_truth() {
        let mut s = StateEstimate::new(vec![0.0, 0.0], vec![vec![100.0, 0.0], vec![0.0, 100.0]]);
        let q = vec![vec![0.01, 0.0], vec![0.0, 0.01]];
        let r = vec![vec![0.5]];
        for i in 1..=15 {
            s = predict(&ConstVel, s, 1.0, &q);
            s = update(&ConstVel, s, &vec![f64::from(i)], &r);
        }
        assert!((s.x[0] - 15.0).abs() < 1.0);
        assert!((s.x[1] - 1.0).abs() < 0.5);
    }

    #[test]
    fn eye_produces_identity() {
        let i = eye(3);
        assert!((i[0][0] - 1.0).abs() < f64::EPSILON);
        assert!((i[1][1] - 1.0).abs() < f64::EPSILON);
        assert!((i[2][2] - 1.0).abs() < f64::EPSILON);
        assert!(i[0][1].abs() < f64::EPSILON);
    }

    #[test]
    fn matmul_identity_returns_original() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let i = eye(2);
        let c = matmul(&a, &i);
        assert_eq!(c, a);
    }

    #[test]
    fn matmul_shape_mismatch_returns_empty() {
        let a = vec![vec![1.0; 3]; 2]; // 2 × 3
        let b = vec![vec![1.0; 2]; 2]; // 2 × 2 (incompatible)
        let c = matmul(&a, &b);
        assert!(c.is_empty());
    }

    #[test]
    fn transpose_swaps_dimensions() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let t = transpose(&a);
        assert_eq!(t.len(), 3);
        assert_eq!(t[0].len(), 2);
        assert!((t[0][0] - 1.0).abs() < f64::EPSILON);
        assert!((t[2][1] - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn invert_recovers_identity_for_2x2() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 1.0]];
        let inv = invert(&a).unwrap();
        let prod = matmul(&a, &inv);
        assert!((prod[0][0] - 1.0).abs() < 1e-9);
        assert!((prod[1][1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn invert_singular_returns_none() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        assert!(invert(&a).is_none());
    }

    #[test]
    fn update_with_singular_r_returns_original_state() {
        let s = StateEstimate::new(vec![0.0, 0.0], vec![vec![0.0; 2]; 2]);
        let s2 = update(&ConstVel, s.clone(), &vec![1.0], &vec![vec![0.0]]);
        assert_eq!(s2.x, s.x);
    }

    #[test]
    fn matvec_returns_correct_dimensions() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let v = vec![1.0, 1.0];
        let r = matvec(&a, &v);
        assert_eq!(r, vec![3.0, 7.0]);
    }
}
