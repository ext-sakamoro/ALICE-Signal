# Changelog

## [Unreleased]

### Changed (破壊的変更: 出力の bit が変わる)

- `fft::fft` / `fft::ifft`: 回転因子を段ごとの漸化式 (`w ← w · w_n`) でなく、1 つずつ角度から `alice_det_math::sin_cos64` で計算する 漸化式は段に沿って丸め誤差が積み上がり、DFT の定義との差が n = 16384 で `ε · log₂ n · ‖x‖` の 100 倍を超えていた (直した後は 0.5 倍未満)
- `windows::{hanning, hamming, blackman}`、`complex::Complex::from_polar`、`fir` の sinc、`iir` の Biquad 係数、`gnss` / `spoofing` の dB 換算、`tracking` の位相: platform の libm でなく `alice-det-math` を使う (target によって最後の bit が違っていた)
- `complex::Complex::mag`: platform の `hypot` でなく、二乗の overflow / underflow を避ける `complex::hypot` を使う

### Added

- `fft::FftPlan` (回転因子の表を 1 回だけ作り、同じ長さの buffer に何度でも使う、`forward` / `inverse`) `fft` / `ifft` は呼ぶたびに plan を作る形になり、出力の bit は変わらない (`tests/fft_plan.rs`、`tests/bits_pinned.rs`)
- `complex::hypot` (IEEE 演算だけで作る `√(a² + b²)`、無限大は NaN より優先)
- `clippy.toml`: platform の超越関数を禁じる (試験の信号生成と独立参照は理由を書いて許可)
- 試験: `tests/fft_accuracy.rs` (DFT の定義との差が `4 · ε · log₂ n · ‖x‖` 以内、参照は補償和)、`tests/bits_pinned.rs` (FFT・窓・絶対値の出力の bit を固定、CI の Linux / macOS / Windows で同じ値)、`tests/hypot_edges.rs`

### Dependencies

- `alice-det-math` 0.5.2 (git rev 2d01c63)
