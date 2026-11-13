# Changelog

## [Unreleased]

### Changed (破壊的変更: 出力の bit が変わる)

- `fft::fft` / `fft::ifft`: 回転因子を段ごとの漸化式 (`w ← w · w_n`) でなく、1 つずつ角度から `alice_det_math::sin_cos64` で計算する 漸化式は段に沿って丸め誤差が積み上がり、DFT の定義との差が n = 16384 で `ε · log₂ n · ‖x‖` の 100 倍を超えていた (直した後は 0.5 倍未満)
- `windows::{hanning, hamming, blackman}`、`complex::Complex::from_polar`、`fir` の sinc、`iir` の Biquad 係数、`gnss` / `spoofing` の dB 換算、`tracking` の位相: platform の libm でなく `alice-det-math` を使う (target によって最後の bit が違っていた)
- `complex::Complex::mag`: platform の `hypot` でなく、二乗の overflow / underflow を避ける `complex::hypot` を使う
- CI: 公開する文面の検査 (`Public text` workflow、`scripts/public_text.sh`) を追加 ALICE-Law の `scripts/public_text_check.py` を `.github/public-text-pin` の revision で取得し、SHA-256 が一致した時だけ使って、HEAD の全 tracked file (path と本文) と送った commit (message・追加行・追加 path) を読む 何も比較できなければ失敗する preflight でも走る

### Added
- CI: 他の crate の `PREVIOUS_*_SEMANTICS_IDS` は一覧全体としてだけ使える (`contains` / `len` / `is_empty`、全体を回す iterator や for、引数として丸ごと渡す) とし、添字・slice のパターン・`.first` / `next` / `take` などを拒む `scripts/upstream_previous_ids_check.py` と試験を足した (ALICE-Edge と同じ中身の共有 file) 上流は major で一覧を空にする 走査した .rs が 0 件なら失敗 (ci.yml と preflight)

- `fft::FftPlan` (回転因子の表を 1 回だけ作り、同じ長さの buffer に何度でも使う、`forward` / `inverse`) `fft` / `ifft` は呼ぶたびに plan を作る形になり、出力の bit は変わらない (`tests/fft_plan.rs`、`tests/bits_pinned.rs`)
- `complex::hypot` (IEEE 演算だけで作る `√(a² + b²)`、無限大は NaN より優先)
- `clippy.toml`: platform の超越関数を禁じる (試験の信号生成と独立参照は理由を書いて許可)
- 試験: `tests/fft_accuracy.rs` (DFT の定義との差が `4 · ε · log₂ n · ‖x‖` 以内、参照は補償和)、`tests/bits_pinned.rs` (FFT・窓・絶対値の出力の bit を固定、CI の Linux / macOS / Windows で同じ値)、`tests/hypot_edges.rs`

### Dependencies

- CI (共有の写し): `scripts/package_integrity.py` と試験 (新規解決は `branch` の git 依存を lock の commit に戻してから照合)、`scripts/release_baseline.py` と試験 (ci/** の版上げは着地後の main と同じ base)、`scripts/test_id_move_check.py` を ALICE-Law 99dd3df4 が計画した bytes にした
- CI: package の検査を共有の `scripts/package_integrity.py` に移した (`branch` で取る git 依存を同じ指定で patch し、展開した crate の Cargo.lock がこの repository の Cargo.lock の commit から取ることを確かめる) それが import する `scripts/alice_siblings_check.py` / `scripts/id_move_check.py` / `scripts/release_baseline.py` と試験、`.github/actions/alice-siblings/action.yml` を共有の bytes で持つ `scripts/package_git_patches.py` と試験は外した (`rev` で取る依存だけを扱っていた)
- `alice-det-math` 0.6 (git の `main`、`Cargo.lock` が commit を名指す)
