//! Discrete-time Kalman filter.
//!
//! Implements the standard time-update / measurement-update recursion for a
//! linear system:
//!
//! ```text
//! x_{k+1} = F x_k + B u_k + w_k     w ~ N(0, Q)
//! z_k     = H x_k + v_k             v ~ N(0, R)
//! ```
//!
//! Matrices are stored row-major in `Vec<f64>` to keep the crate pure Rust
//! without any linear-algebra dependency. Dimensions are fixed at
//! construction time; runtime shape checks return `None` on mismatch.
//!
//! For `GNSS` receivers this is the workhorse behind Position-Velocity-Time
//! (`PVT`) fusion and code-carrier smoothing; a spoofing-detection tap
//! typically observes the innovation sequence produced by
//! [`KalmanFilter::innovation`].

// ---------------------------------------------------------------------------
// Matrix helpers
// ---------------------------------------------------------------------------

/// Integer square root; returns `Some(m)` iff `x == m * m`.
fn integer_sqrt(x: usize) -> Option<usize> {
    if x == 0 {
        return Some(0);
    }
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let mut m = (x as f64).sqrt() as usize;
    // Adjust for rounding.
    while m * m > x {
        m -= 1;
    }
    while (m + 1) * (m + 1) <= x {
        m += 1;
    }
    if m * m == x {
        Some(m)
    } else {
        None
    }
}

fn zeros(rows: usize, cols: usize) -> Vec<f64> {
    vec![0.0; rows * cols]
}

fn identity(n: usize) -> Vec<f64> {
    let mut m = zeros(n, n);
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

fn matmul(a: &[f64], a_rows: usize, a_cols: usize, b: &[f64], b_cols: usize) -> Vec<f64> {
    let mut out = zeros(a_rows, b_cols);
    for i in 0..a_rows {
        for j in 0..b_cols {
            let mut sum = 0.0;
            for k in 0..a_cols {
                sum += a[i * a_cols + k] * b[k * b_cols + j];
            }
            out[i * b_cols + j] = sum;
        }
    }
    out
}

fn transpose(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = zeros(cols, rows);
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = a[i * cols + j];
        }
    }
    out
}

fn matadd(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

fn matsub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

/// Invert a general square matrix via Gauss-Jordan with partial pivoting.
fn invert(m: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut a = vec![0.0; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            a[i * 2 * n + j] = m[i * n + j];
        }
        a[i * 2 * n + n + i] = 1.0;
    }
    for i in 0..n {
        // Partial pivot.
        let mut pivot_row = i;
        let mut pivot_val = a[i * 2 * n + i].abs();
        for r in (i + 1)..n {
            let v = a[r * 2 * n + i].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = r;
            }
        }
        if pivot_val < 1e-14 {
            return None;
        }
        if pivot_row != i {
            for c in 0..2 * n {
                a.swap(i * 2 * n + c, pivot_row * 2 * n + c);
            }
        }
        // Scale pivot row.
        let pivot = a[i * 2 * n + i];
        for c in 0..2 * n {
            a[i * 2 * n + c] /= pivot;
        }
        // Eliminate other rows.
        for r in 0..n {
            if r == i {
                continue;
            }
            let factor = a[r * 2 * n + i];
            for c in 0..2 * n {
                a[r * 2 * n + c] -= factor * a[i * 2 * n + c];
            }
        }
    }
    let mut inv = zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = a[i * 2 * n + n + j];
        }
    }
    Some(inv)
}

// ---------------------------------------------------------------------------
// KalmanFilter
// ---------------------------------------------------------------------------

/// Fixed-dimension discrete-time Kalman filter with `n`-dimensional state
/// and `m`-dimensional measurement.
#[derive(Debug, Clone)]
pub struct KalmanFilter {
    n: usize,
    m: usize,
    /// State vector, length `n`.
    x: Vec<f64>,
    /// Covariance, `n × n`.
    p: Vec<f64>,
    /// State transition, `n × n`.
    f: Vec<f64>,
    /// Process noise, `n × n`.
    q: Vec<f64>,
    /// Measurement model, `m × n`.
    h: Vec<f64>,
    /// Measurement noise, `m × m`.
    r: Vec<f64>,
}

impl KalmanFilter {
    /// Construct a filter with the given matrices.  All matrices are stored
    /// row-major.  Returns `None` when any matrix has the wrong shape.
    #[must_use]
    pub fn new(
        x0: Vec<f64>,
        p0: Vec<f64>,
        f: Vec<f64>,
        q: Vec<f64>,
        h: Vec<f64>,
        r: Vec<f64>,
    ) -> Option<Self> {
        let n = x0.len();
        if p0.len() != n * n || f.len() != n * n || q.len() != n * n {
            return None;
        }
        let m = integer_sqrt(r.len())?;
        if h.len() != m * n {
            return None;
        }
        Some(Self {
            n,
            m,
            x: x0,
            p: p0,
            f,
            q,
            h,
            r,
        })
    }

    /// State dimension.
    #[must_use]
    pub const fn state_dim(&self) -> usize {
        self.n
    }

    /// Measurement dimension.
    #[must_use]
    pub const fn measurement_dim(&self) -> usize {
        self.m
    }

    /// Current a-posteriori state estimate.
    #[must_use]
    pub fn state(&self) -> &[f64] {
        &self.x
    }

    /// Current a-posteriori covariance.
    #[must_use]
    pub fn covariance(&self) -> &[f64] {
        &self.p
    }

    /// Time update (prediction) step.
    pub fn predict(&mut self) {
        // x = F x
        let new_x = matmul(&self.f, self.n, self.n, &self.x, 1);
        // P = F P F^T + Q
        let fp = matmul(&self.f, self.n, self.n, &self.p, self.n);
        let ft = transpose(&self.f, self.n, self.n);
        let fpft = matmul(&fp, self.n, self.n, &ft, self.n);
        self.p = matadd(&fpft, &self.q);
        self.x = new_x;
    }

    /// Measurement update step.
    ///
    /// Returns `None` when `z` has the wrong length or when the innovation
    /// covariance is singular.
    pub fn update(&mut self, z: &[f64]) -> Option<()> {
        if z.len() != self.m {
            return None;
        }
        // y = z - H x
        let hx = matmul(&self.h, self.m, self.n, &self.x, 1);
        let y = matsub(z, &hx);
        // S = H P H^T + R
        let hp = matmul(&self.h, self.m, self.n, &self.p, self.n);
        let ht = transpose(&self.h, self.m, self.n);
        let hpht = matmul(&hp, self.m, self.n, &ht, self.m);
        let s = matadd(&hpht, &self.r);
        let s_inv = invert(&s, self.m)?;
        // K = P H^T S^-1
        let pht = matmul(&self.p, self.n, self.n, &ht, self.m);
        let k = matmul(&pht, self.n, self.m, &s_inv, self.m);
        // x = x + K y
        let ky = matmul(&k, self.n, self.m, &y, 1);
        self.x = matadd(&self.x, &ky);
        // P = (I - K H) P
        let kh = matmul(&k, self.n, self.m, &self.h, self.n);
        let ikh = matsub(&identity(self.n), &kh);
        self.p = matmul(&ikh, self.n, self.n, &self.p, self.n);
        Some(())
    }

    /// Compute the innovation `y = z - H x` under the current predicted state.
    ///
    /// A spoofing / jamming monitor typically thresholds this sequence.
    #[must_use]
    pub fn innovation(&self, z: &[f64]) -> Option<Vec<f64>> {
        if z.len() != self.m {
            return None;
        }
        let hx = matmul(&self.h, self.m, self.n, &self.x, 1);
        Some(matsub(z, &hx))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn constant_velocity_filter() -> KalmanFilter {
        // State: [position, velocity].
        // dt = 1 s, measurement observes position only.
        let x0 = vec![0.0, 0.0];
        let p0 = vec![10.0, 0.0, 0.0, 10.0];
        let f = vec![1.0, 1.0, 0.0, 1.0];
        let q = vec![0.01, 0.0, 0.0, 0.01];
        let h = vec![1.0, 0.0];
        let r = vec![1.0];
        KalmanFilter::new(x0, p0, f, q, h, r).expect("valid shapes")
    }

    #[test]
    fn constructor_rejects_wrong_shapes() {
        let x0 = vec![0.0, 0.0];
        let bad_p = vec![1.0, 0.0, 0.0]; // wrong length
        let f = vec![1.0, 0.0, 0.0, 1.0];
        let q = vec![0.0; 4];
        let h = vec![1.0, 0.0];
        let r = vec![1.0];
        assert!(KalmanFilter::new(x0, bad_p, f, q, h, r).is_none());
    }

    #[test]
    fn predict_advances_state_by_transition() {
        let mut kf = constant_velocity_filter();
        // Inject an initial velocity of 1.0.
        kf.x = vec![0.0, 1.0];
        kf.predict();
        assert!(approx(kf.state()[0], 1.0, 1e-9));
        assert!(approx(kf.state()[1], 1.0, 1e-9));
    }

    #[test]
    fn update_pulls_state_toward_measurement() {
        let mut kf = constant_velocity_filter();
        // Predict once, then correct with a large position measurement.
        kf.predict();
        kf.update(&[5.0]).unwrap();
        assert!(kf.state()[0] > 2.0);
    }

    #[test]
    fn innovation_is_z_minus_hx() {
        let kf = constant_velocity_filter();
        let innov = kf.innovation(&[3.0]).unwrap();
        assert_eq!(innov.len(), 1);
        // Initial state is zero so innovation = z.
        assert!(approx(innov[0], 3.0, 1e-9));
    }

    #[test]
    fn steady_state_converges_toward_true_velocity() {
        // Simulate a target moving at constant velocity v = 2 with
        // position observations corrupted by white noise (deterministic
        // pseudo-random sequence).
        let mut kf = constant_velocity_filter();
        let true_v = 2.0;
        let mut true_p = 0.0;
        // Fixed sequence of ±0.5 as pseudo-noise.
        let noise = [0.5_f64, -0.3, 0.2, -0.4, 0.1, 0.3, -0.5, 0.4, -0.1, 0.2];
        for &n in &noise {
            true_p += true_v;
            kf.predict();
            kf.update(&[true_p + n]).unwrap();
        }
        // Estimated velocity should be close to 2.
        assert!(
            (kf.state()[1] - true_v).abs() < 0.5,
            "estimated velocity: {}",
            kf.state()[1]
        );
    }

    #[test]
    fn update_rejects_wrong_measurement_size() {
        let mut kf = constant_velocity_filter();
        assert!(kf.update(&[1.0, 2.0]).is_none());
    }

    #[test]
    fn singular_measurement_matrix_returns_none() {
        // R = 0 and P = 0 gives singular S in the update.
        let x0 = vec![0.0];
        let p0 = vec![0.0];
        let f = vec![1.0];
        let q = vec![0.0];
        let h = vec![1.0];
        let r = vec![0.0];
        let mut kf = KalmanFilter::new(x0, p0, f, q, h, r).unwrap();
        assert!(kf.update(&[1.0]).is_none());
    }

    #[test]
    fn invert_identity_returns_identity() {
        let m = identity(3);
        let inv = invert(&m, 3).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx(inv[i * 3 + j], expected, 1e-12));
            }
        }
    }

    #[test]
    fn invert_singular_returns_none() {
        let singular = vec![1.0, 2.0, 2.0, 4.0];
        assert!(invert(&singular, 2).is_none());
    }
}
