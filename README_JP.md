[English](README.md) | **日本語**

# ALICE-Signal

ALICEエコシステム向けの純Rustデジタル信号処理ライブラリ。外部依存なし。

## 機能

- **FFT** -- Cooley-Tukeyラディックス2 FFTおよび逆FFT
- **FIRフィルタ** -- ローパス・ハイパスフィルタの窓関数sinc設計
- **IIRフィルタ** -- Biquadフィルタ(ローパス、ハイパス、バンドパス)、カットオフ・Q値設定可能
- **ウェーブレット変換** -- Haar(単一レベル・多レベル)、Daubechies-4(順変換/逆変換)
- **窓関数** -- ハミング、ハニング、ブラックマン
- **畳み込み・相関** -- 線形畳み込みと相互相関
- **スペクトル解析** -- パワースペクトル密度(PSD)、窓関数付きオプション
- **リサンプリング** -- デシメーション、ゼロ挿入補間、線形補間
- **ユーティリティ** -- エネルギー、RMS、2のべき乗ゼロパディング、複素数型

## アーキテクチャ

```
Complex (最小複素数型)
    |
    v
FFT / IFFT (Cooley-Tukeyラディックス2)
    |
    +-- PSD (パワースペクトル密度)
    +-- 窓関数 (ハミング, ハニング, ブラックマン)
    |
    v
FIR (窓関数sinc)   IIR (Biquad)
    |                    |
    v                    v
畳み込み / 相関
    |
    v
ウェーブレット (Haar, Daubechies-4)
    |
    v
デシメーション / 補間
```

## クイックスタート

```rust
use alice_signal::*;

// FFT
let mut buf = vec![Complex::new(1.0, 0.0); 8];
fft(&mut buf);

// FIRローパスフィルタ
let coeffs = fir_lowpass(31, 0.25);
let filtered = fir_filter(&signal, &coeffs);

// Biquad IIRフィルタ
let bq = Biquad::lowpass(0.1, 0.707);
let output = bq.filter(&signal);

// Haarウェーブレット変換
haar_forward_multi(&mut data);
```

## ライセンス

MIT OR Apache-2.0
