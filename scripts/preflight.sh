#!/usr/bin/env bash
# scripts/preflight.sh — local reproduction of the CI gates before `git push`.
# Every command is the one CI runs. A step this file does not cover is a step
# that can only fail remotely — when a workflow changes, regenerate this file
# in the same commit (checks no workflow runs go in scripts/preflight.local.sh).
# Run `--quick` before every push.
#
# usage: scripts/preflight.sh [--quick]   (--quick skips the test / bench suites)
# generator rev 86c79aeadabd0baa7445f75f8a0df4bb076a74cb
set -euo pipefail
cd "$(dirname "$0")/.."

quick=0
[[ "${1:-}" == "--quick" ]] && quick=1

step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
# `cargo clippy` reuses fresh `cargo check` artifacts and then lints nothing;
# touching the crate roots invalidates only this repo's fingerprints.
relint() { git ls-files | grep -E '(^|/)src/(lib|main)\.rs$' | xargs -r touch; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "missing tool: $1 ($2)" >&2; exit 1; }; }
has_toolchain() { rustup toolchain list | grep -q "^$1"; }

# Steps CI runs that this file cannot reproduce locally (they can only fail remotely):
#   - security-audit.yml:audit:Install cargo-audit (needs network / runner-only)
#   - security-audit.yml:deny:Install cargo-deny (needs network / runner-only)
#   - security-audit.yml:coverage (job is continue-on-error: informational in CI)
#   - security-audit.yml:semver-checks (job is continue-on-error: informational in CI)
#   - fuzz.yml:fuzz:Install cargo-fuzz [target=fuzz_dsp] (needs network / runner-only)
#   - fuzz.yml:fuzz:Set fuzz duration [target=fuzz_dsp] (no cargo / grep)
#   - fuzz.yml:fuzz:Run fuzz target (time-boxed) [target=fuzz_dsp] (continue-on-error)
#   - fuzz.yml:fuzz:Report crash (informational) [target=fuzz_dsp] (no cargo / grep)
#   - fuzz.yml:fuzz:Install cargo-fuzz [target=fuzz_gnss] (needs network / runner-only)
#   - fuzz.yml:fuzz:Set fuzz duration [target=fuzz_gnss] (no cargo / grep)
#   - fuzz.yml:fuzz:Run fuzz target (time-boxed) [target=fuzz_gnss] (continue-on-error)
#   - fuzz.yml:fuzz:Report crash (informational) [target=fuzz_gnss] (no cargo / grep)
#   - fuzz.yml:fuzz:Install cargo-fuzz [target=fuzz_kalman] (needs network / runner-only)
#   - fuzz.yml:fuzz:Set fuzz duration [target=fuzz_kalman] (no cargo / grep)
#   - fuzz.yml:fuzz:Run fuzz target (time-boxed) [target=fuzz_kalman] (continue-on-error)
#   - fuzz.yml:fuzz:Report crash (informational) [target=fuzz_kalman] (no cargo / grep)

need actionlint "brew install actionlint"
need cargo-audit "cargo install cargo-audit --locked"
need cargo-deny "cargo install cargo-deny --locked"
need cargo-machete "cargo install cargo-machete --locked"
has_toolchain 1.87 || { echo "missing toolchain 1.87 (rustup toolchain install 1.87)" >&2; exit 1; }

step "ci.yml / fmt: run"
( export CARGO_TERM_COLOR="always" CARGO_INCREMENTAL="0" RUST_BACKTRACE="short"; cargo fmt --all -- --check )

step "ci.yml / clippy: run"
relint
( export CARGO_TERM_COLOR="always" CARGO_INCREMENTAL="0" RUST_BACKTRACE="short"; cargo clippy --all-targets -- -D warnings )

step "ci.yml / msrv: run"
( export CARGO_TERM_COLOR="always" CARGO_INCREMENTAL="0" RUST_BACKTRACE="short"; cargo +1.87 check --lib )

step "ci.yml / doc: run"
( export CARGO_TERM_COLOR="always" CARGO_INCREMENTAL="0" RUST_BACKTRACE="short" RUSTDOCFLAGS="-Dwarnings"; cargo doc --no-deps )

step "ci.yml / actionlint: actionlint"
actionlint .github/workflows/*.yml

step "security-audit.yml / deny: Run cargo deny check all"
( export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"; cargo deny --all-features check all )

step "security-audit.yml / unused-deps: cargo machete"
cargo machete

step "security-audit.yml / stub-guard: Block panic!(STUB) in src/**"
(
  export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"
  set -eo pipefail
  hits=$(grep -rnE 'panic!\([^)]*STUB' \
    src/ --include="*.rs" \
    --exclude-dir=bin \
    | grep -v ':[[:space:]]*//' \
    | grep -vE ':[[:space:]]*/\*' \
    || true)
  if [ -n "$hits" ]; then
    echo "❌ Explicit STUB panic detected in src/ (production path):"
    echo "$hits"
    exit 1
  fi
  echo "✓ No panic!(STUB) in src/"
)

step "security-audit.yml / stub-guard: Detect todo! / unimplemented! (informational, not blocking)"
(
  export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"
  set -eo pipefail
  hits=$(grep -rnE 'todo!\(|unimplemented!\(' \
    src/ --include="*.rs" \
    --exclude-dir=bin \
    | grep -v ':[[:space:]]*//' \
    | grep -vE ':[[:space:]]*/\*' \
    || true)
  if [ -n "$hits" ]; then
    count=$(echo "$hits" | wc -l | tr -d ' ')
    echo "::warning::${count} todo!()/unimplemented!() marker(s) in src/ (informational, a marked stub is a legitimate fail-fast idiom):"
    echo "$hits" | head -20
  else
    echo "✓ No todo!/unimplemented! markers in src/"
  fi
)

step "security-audit.yml / stub-guard: Detect dbg!() residual in src/**"
(
  export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"
  set -eo pipefail
  hits=$(grep -rn 'dbg!(' src/ --include="*.rs" || true)
  if [ -n "$hits" ]; then
    echo "❌ dbg!() macro left in src/:"
    echo "$hits"
    exit 1
  fi
  echo "✓ No dbg!() in src/"
)

step "security-audit.yml / stub-guard: Detect TODO / FIXME / XXX / HACK (informational)"
(
  export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"
  set -eo pipefail
  hits=$(grep -rnE 'TODO|FIXME|XXX|HACK' src/ --include="*.rs" || true)
  if [ -n "$hits" ]; then
    echo "::warning::TODO/FIXME/XXX/HACK found in src/ (informational, not blocking):"
    echo "$hits" | head -50
  else
    echo "✓ No TODO/FIXME/XXX/HACK in src/"
  fi
)

step "security-audit.yml / package-integrity: cargo package (.crate を作る)"
( export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"; cargo package --no-verify --allow-dirty )

step "security-audit.yml / package-integrity: 展開して隔離 build"
(
  export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"
  set -euo pipefail
  crate=$(find "${CARGO_TARGET_DIR:-target}/package" -maxdepth 1 -name '*.crate' | head -1)
  [ -n "$crate" ] || { echo "::error::.crate が生成されていない"; exit 1; }
  echo "packaged: $(basename "$crate") ($(du -h "$crate" | cut -f1))"
  tmp=$(mktemp -d)
  tar xzf "$crate" -C "$tmp"
  cd "$tmp"/*/
  # 親 workspace を継承すると member 扱いになり、境界を越えた file が
  # 再び見えてしまう (検査の意味が消える) ので単独 package として build
  printf '\n[workspace]\n' >> Cargo.toml
  cargo check --all-features
)

step "fuzz.yml / fuzz: Build fuzz target [target=fuzz_dsp]"
if has_toolchain nightly && cargo +nightly fuzz --version >/dev/null 2>&1; then
  (
    export CARGO_TERM_COLOR="always"
    cd fuzz
    cargo +nightly fuzz build "fuzz_dsp"
  )
else
  echo "skip: nightly / cargo-fuzz not installed" >&2
fi

step "fuzz.yml / fuzz: Build fuzz target [target=fuzz_gnss]"
if has_toolchain nightly && cargo +nightly fuzz --version >/dev/null 2>&1; then
  (
    export CARGO_TERM_COLOR="always"
    cd fuzz
    cargo +nightly fuzz build "fuzz_gnss"
  )
else
  echo "skip: nightly / cargo-fuzz not installed" >&2
fi

step "fuzz.yml / fuzz: Build fuzz target [target=fuzz_kalman]"
if has_toolchain nightly && cargo +nightly fuzz --version >/dev/null 2>&1; then
  (
    export CARGO_TERM_COLOR="always"
    cd fuzz
    cargo +nightly fuzz build "fuzz_kalman"
  )
else
  echo "skip: nightly / cargo-fuzz not installed" >&2
fi

if [[ $quick -eq 1 ]]; then
  echo; echo "preflight --quick OK (test / bench suites skipped)"; exit 0
fi

step "ci.yml / test: Test"
( export CARGO_TERM_COLOR="always" CARGO_INCREMENTAL="0" RUST_BACKTRACE="short"; cargo test )

step "ci.yml / test: Analytic oracle"
( export CARGO_TERM_COLOR="always" CARGO_INCREMENTAL="0" RUST_BACKTRACE="short"; cargo test --test analytic_oracle )

step "ci.yml / test: Doc tests"
( export CARGO_TERM_COLOR="always" CARGO_INCREMENTAL="0" RUST_BACKTRACE="short"; cargo test --doc )

step "ci.yml / test-public-siblings: Test (lib, integration, doctests)"
(
  export CARGO_TERM_COLOR="always" CARGO_INCREMENTAL="0" RUST_BACKTRACE="short" MIN_PASSED="207"
  # the default shell has no pipefail: without it cargo's exit status is lost in the pipe
  set -o pipefail
  log="$(mktemp)"
  cargo test --all-features 2>&1 | tee "${log}"
  passed=$(grep -E '^test result: ' "${log}" | awk '{s += $4} END {print s + 0}')
  failed=$(grep -E '^test result: ' "${log}" | awk '{s += $6} END {print s + 0}')
  echo "passed: ${passed} (minimum ${MIN_PASSED}), failed: ${failed}"
  if [ "${failed}" -ne 0 ] || grep -E '^test result: ' "${log}" | grep -vE '^test result: ok\. ' >/dev/null; then
    echo "::error::${failed} tests failed"
    exit 1
  fi
  if [ "${passed}" -lt "${MIN_PASSED}" ]; then
    echo "::error::${passed} tests passed, fewer than ${MIN_PASSED}"
    exit 1
  fi
)

step "security-audit.yml / audit: Run cargo audit"
(
  export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"
  cargo audit --deny yanked \
    --ignore RUSTSEC-2024-0436 \
    --ignore RUSTSEC-2025-0020 \
    --ignore RUSTSEC-2026-0176 \
    --ignore RUSTSEC-2026-0177 \
    --ignore RUSTSEC-2026-0204 \
    --ignore RUSTSEC-2026-0235
)

echo; echo "preflight OK"
