**English** | [日本語](README_JP.md)

# ALICE-Signal

Pure Rust digital signal processing library for the ALICE ecosystem. Zero external dependencies.

## Features

- **FFT** -- Cooley-Tukey radix-2 FFT and inverse FFT
- **FIR Filters** -- Windowed-sinc design for lowpass and highpass filters
- **IIR Filters** -- Biquad filters (lowpass, highpass, bandpass) with configurable cutoff and Q factor
- **Wavelet Transforms** -- Haar (single-level and multi-level) and Daubechies-4 (forward/inverse)
- **Window Functions** -- Hamming, Hanning, Blackman
- **Convolution & Correlation** -- Linear convolution and cross-correlation
- **Spectral Analysis** -- Power spectral density (PSD) with optional windowing
- **Resampling** -- Decimation, zero-insertion interpolation, linear interpolation
- **Utilities** -- Energy, RMS, zero-padding to power-of-two, complex number type

## Architecture

```
Complex (minimal complex number)
    |
    v
FFT / IFFT (Cooley-Tukey radix-2)
    |
    +-- PSD (power spectral density)
    +-- Window functions (Hamming, Hanning, Blackman)
    |
    v
FIR (windowed sinc)   IIR (Biquad)
    |                      |
    v                      v
Convolution / Correlation
    |
    v
Wavelet (Haar, Daubechies-4)
    |
    v
Decimation / Interpolation
```

## Quick Start

```rust
use alice_signal::*;

// FFT
let mut buf = vec![Complex::new(1.0, 0.0); 8];
fft(&mut buf);

// FIR lowpass filter
let coeffs = fir_lowpass(31, 0.25);
let filtered = fir_filter(&signal, &coeffs);

// Biquad IIR filter
let bq = Biquad::lowpass(0.1, 0.707);
let output = bq.filter(&signal);

// Haar wavelet transform
haar_forward_multi(&mut data);
```

## License

MIT OR Apache-2.0
