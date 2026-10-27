//! complex.

// ---------------------------------------------------------------------------
// Complex number (minimal, no external deps)
// ---------------------------------------------------------------------------

/// Minimal complex number for FFT operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    #[must_use]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[must_use]
    pub fn mag(self) -> f64 {
        hypot(self.re, self.im)
    }

    #[must_use]
    pub fn mag_sq(self) -> f64 {
        self.re.mul_add(self.re, self.im * self.im)
    }

    #[must_use]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    #[must_use]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * alice_det_math::cos64(theta),
            im: r * alice_det_math::sin64(theta),
        }
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re.mul_add(rhs.re, -(self.im * rhs.im)),
            im: self.re.mul_add(rhs.im, self.im * rhs.re),
        }
    }
}

impl std::ops::Mul<f64> for Complex {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

/// `√(a² + b²)` without overflow or underflow in the intermediate squares, from IEEE operations
/// only (the same bits on every target, unlike the platform `hypot`).
#[must_use]
pub fn hypot(a: f64, b: f64) -> f64 {
    let (x, y) = (a.abs(), b.abs());
    // IEEE 754 hypot: an infinity wins over a NaN
    if x.is_infinite() || y.is_infinite() {
        return f64::INFINITY;
    }
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    let (big, small) = if x >= y { (x, y) } else { (y, x) };
    if big == 0.0 {
        return 0.0;
    }
    let r = small / big;
    big * r.mul_add(r, 1.0).sqrt()
}
