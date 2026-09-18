//! Fuzz: arbitrary PRN numbers, I/Q prompt streams and ±1 chip sequences
//! through the GNSS layer — `ca_code`, `estimate_cn0` (any window config),
//! `normalised_correlation`, the E/P/L correlator at any code offset, the
//! PLL / FLL discriminators + loop filter and the spoofing detector. None may
//! panic (division by zero, empty windows, out-of-range offsets).
#![no_main]

use alice_signal::correlator::{make_ca_correlator, Correlator};
use alice_signal::gnss::{ca_code, estimate_cn0, normalised_correlation, Cn0Config};
use alice_signal::spoofing::{
    clock_jump_score, cn0_anomaly_score, position_jump_score, satellite_count_score, Detector,
};
use alice_signal::tracking::{
    fll_cross_product_discriminator, pll_costas_discriminator, SecondOrderLoop,
};
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    prn: u16,
    i: Vec<f64>,
    q: Vec<f64>,
    chips: Vec<i8>,
    window: u8,
    windows: u8,
    t: f64,
    offset: u16,
    epochs: Vec<(f64, f64, f64, [f64; 3], u8)>,
}

fuzz_target!(|input: Input| {
    let code = ca_code(input.prn);
    if let Some(c) = &code {
        assert_eq!(c.len(), 1023);
    }
    let n = input.i.len().min(input.q.len()).min(4096);
    let (i, q) = (&input.i[..n], &input.q[..n]);
    let cfg = Cn0Config {
        samples_per_window: usize::from(input.window),
        windows: usize::from(input.windows),
        coherent_integration_s: input.t,
    };
    let _ = estimate_cn0(i, q, cfg);
    let _ = estimate_cn0(i, q, Cn0Config::default());

    let chips: Vec<i8> = input.chips.iter().take(2048).map(|c| if *c < 0 { -1 } else { 1 }).collect();
    let _ = normalised_correlation(&chips, &chips);
    if let Some(c) = code {
        let _ = normalised_correlation(&c, &chips);
        let corr = make_ca_correlator(c);
        let _ = corr.correlate(i, usize::from(input.offset));
    }
    if !chips.is_empty() {
        // documented contract: half_chip_offset > 0
        let corr = Correlator::new(chips.clone(), usize::from(input.window) + 1);
        let _ = corr.code_len();
        if let Some(out) = corr.correlate(i, usize::from(input.offset) % chips.len()) {
            let _ = (out.early_minus_late(), out.normalised_early_minus_late(), out.prompt_power());
        }
    }

    let _ = pll_costas_discriminator(input.t, input.t.sin());
    let _ = fll_cross_product_discriminator(1.0, input.t, input.t, 1.0, input.t);
    let mut lf = SecondOrderLoop::new(input.t.abs(), 0.707, 1e-3);
    for v in i.iter().take(64) {
        let _ = lf.update(*v);
    }

    let _ = (cn0_anomaly_score(input.t), clock_jump_score(input.t, input.t), position_jump_score(input.t, input.t, input.t), satellite_count_score(usize::from(input.window)));
    let mut det = Detector::default();
    for (epoch, cn0, bias, pos, sats) in input.epochs.iter().take(64) {
        let _ = det.push(*epoch, *cn0, *bias, *pos, usize::from(*sats));
    }
    det.reset();
});
