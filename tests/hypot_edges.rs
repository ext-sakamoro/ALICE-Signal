//! Oracle: `complex::hypot` follows IEEE 754 `hypot` on the cases that a naive `√(a² + b²)` gets
//! wrong (overflow and underflow of the squares, an infinity together with a NaN), and is exact on
//! Pythagorean triples.

use alice_signal::complex::{hypot, Complex};

#[test]
fn exact_on_triples_and_symmetric() {
    for (a, b, c) in [
        (3.0, 4.0, 5.0),
        (5.0, 12.0, 13.0),
        (8.0, 15.0, 17.0),
        (0.0, 7.0, 7.0),
    ] {
        assert_eq!(hypot(a, b), c);
        assert_eq!(hypot(b, a), c);
        assert_eq!(hypot(-a, b), c);
        assert_eq!(hypot(a, -b), c);
    }
    assert_eq!(hypot(0.0, 0.0), 0.0);
    assert_eq!(hypot(-0.0, 0.0).to_bits(), 0.0_f64.to_bits());
    assert_eq!(Complex::new(3.0, -4.0).mag(), 5.0);
}

#[test]
fn no_overflow_or_underflow_in_the_squares() {
    // a² overflows, the result does not: √2 · 1e300
    let big = hypot(1e300, 1e300);
    assert!(big.is_finite());
    assert!((big / 1e300 - std::f64::consts::SQRT_2).abs() <= 2.0 * f64::EPSILON);
    // a² underflows to 0, the result does not: √2 · 1e-300
    let small = hypot(1e-300, 1e-300);
    assert!(small > 0.0);
    assert!((small / 1e-300 - std::f64::consts::SQRT_2).abs() <= 2.0 * f64::EPSILON);
    assert_eq!(hypot(f64::MAX, 0.0), f64::MAX);
}

#[test]
fn infinity_wins_over_nan() {
    assert_eq!(hypot(f64::INFINITY, f64::NAN), f64::INFINITY);
    assert_eq!(hypot(f64::NAN, f64::NEG_INFINITY), f64::INFINITY);
    assert_eq!(hypot(f64::INFINITY, 1.0), f64::INFINITY);
    assert!(hypot(f64::NAN, 1.0).is_nan());
    assert!(hypot(0.0, f64::NAN).is_nan());
    assert!(hypot(f64::NAN, 0.0).is_nan());
}
