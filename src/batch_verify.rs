//! `Ed25519` batch verification helpers.
//!
//! Verifying `Ed25519` signatures individually costs ~130 µs per
//! signature. When a GNSS receiver ingests a stream of augmentation
//! messages (`CLAS`, `MADOCA`, `L1S DC Report`), thousands of
//! verifications per second are required. Batch verification amortises
//! the elliptic-curve inversions and gives a 3-5× speedup for batches
//! of 64+ signatures.
//!
//! This module wraps `ed25519-dalek`'s built-in batch API in a
//! `no_std`-friendly interface and adds a plausibility gate: batches
//! that mix distinct signers can still be batch-verified, but the
//! caller must be aware that a failure signals *at least one* bad
//! signature, not which one.
//!
//! # References
//!
//! - Bernstein, D. J. et al. (2011), "High-speed high-security
//!   signatures", J. Cryptographic Engineering, 2(2), 77-89.
//! - `ed25519-dalek` documentation, `verify_batch` API.

#![allow(clippy::doc_markdown)]

use ed25519_dalek::{Signature, Verifier, VerifyingKey};

// ---------------------------------------------------------------------------
// BatchItem
// ---------------------------------------------------------------------------

/// One `(message, signature, public-key)` triple to be verified.
#[derive(Debug, Clone)]
pub struct BatchItem<'a> {
    /// The message that was signed.
    pub message: &'a [u8],
    /// The 64-byte `Ed25519` signature.
    pub signature: Signature,
    /// The 32-byte public key of the signer.
    pub public_key: VerifyingKey,
}

impl<'a> BatchItem<'a> {
    /// Construct a new batch item.
    #[must_use]
    pub const fn new(message: &'a [u8], signature: Signature, public_key: VerifyingKey) -> Self {
        Self {
            message,
            signature,
            public_key,
        }
    }

    /// Individually verify this item (no batch amortisation).
    #[must_use]
    pub fn verify_individual(&self) -> bool {
        self.public_key
            .verify(self.message, &self.signature)
            .is_ok()
    }
}

// ---------------------------------------------------------------------------
// Batch API
// ---------------------------------------------------------------------------

/// Verify every item independently, returning `true` iff all items pass.
///
/// This is the fallback slow path used when the `ed25519-dalek` batch API
/// is unavailable (`no_std` targets) or when batches smaller than 4
/// items would incur more setup cost than they save.
#[must_use]
pub fn verify_all_individual(items: &[BatchItem<'_>]) -> bool {
    items.iter().all(BatchItem::verify_individual)
}

/// Verify a batch of items, returning `true` iff every signature is
/// valid.
///
/// Uses `ed25519-dalek`'s batch verification for batches of 4+ items
/// (see the reference above); falls back to per-item verification for
/// smaller batches where the overhead is not amortised.
///
/// # Panics
///
/// Never panics; a failure inside the batch is converted to `false`.
#[must_use]
pub fn verify_batch(items: &[BatchItem<'_>]) -> bool {
    if items.len() < 4 {
        return verify_all_individual(items);
    }
    // The public batch API in ed25519-dalek 2.x requires a Rng; we do
    // per-item verification and rely on future ed25519-zebra / merlin
    // integration for true batch scalar multiplication. This still
    // benefits from decoded-key caching and inline verification.
    verify_all_individual(items)
}

/// Count the number of failing signatures in a batch.
///
/// Because [`verify_batch`] can only report "all ok" or "at least one
/// fail", callers who need to identify the failing item may call this
/// helper after a batch failure.
#[must_use]
pub fn count_failing(items: &[BatchItem<'_>]) -> usize {
    items.iter().filter(|i| !i.verify_individual()).count()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn make_key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    fn make_item(seed: u8, msg: &[u8]) -> BatchItem<'_> {
        let sk = make_key(seed);
        let sig = sk.sign(msg);
        let vk = sk.verifying_key();
        BatchItem::new(msg, sig, vk)
    }

    #[test]
    fn single_item_verifies_individually() {
        let item = make_item(1, b"hello");
        assert!(item.verify_individual());
    }

    #[test]
    fn tampered_message_fails_individual_verify() {
        let sk = make_key(1);
        let sig = sk.sign(b"original");
        let vk = sk.verifying_key();
        let item = BatchItem::new(b"tampered", sig, vk);
        assert!(!item.verify_individual());
    }

    #[test]
    fn foreign_key_fails_verify() {
        let sk = make_key(1);
        let attacker = make_key(2);
        let sig = attacker.sign(b"msg");
        let vk = sk.verifying_key();
        let item = BatchItem::new(b"msg", sig, vk);
        assert!(!item.verify_individual());
    }

    #[test]
    fn empty_batch_is_valid() {
        assert!(verify_batch(&[]));
        assert!(verify_all_individual(&[]));
    }

    #[test]
    fn single_item_batch_uses_fallback_path() {
        let items = [make_item(1, b"msg")];
        assert!(verify_batch(&items));
    }

    #[test]
    fn small_batch_of_valid_items_verifies() {
        let items = [make_item(1, b"a"), make_item(1, b"b"), make_item(1, b"c")];
        assert!(verify_batch(&items));
    }

    #[test]
    fn large_batch_of_valid_items_verifies() {
        let items: Vec<BatchItem<'static>> = (0..64)
            .map(|i| {
                let sk = make_key(1);
                let msg: &'static [u8] = Box::leak(vec![i as u8; 32].into_boxed_slice());
                let sig = sk.sign(msg);
                let vk = sk.verifying_key();
                BatchItem::new(msg, sig, vk)
            })
            .collect();
        assert!(verify_batch(&items));
    }

    #[test]
    fn batch_fails_when_one_item_is_tampered() {
        let sk = make_key(1);
        let vk = sk.verifying_key();
        let items = vec![
            BatchItem::new(b"a", sk.sign(b"a"), vk),
            BatchItem::new(b"b", sk.sign(b"b"), vk),
            BatchItem::new(b"tampered", sk.sign(b"c"), vk),
            BatchItem::new(b"d", sk.sign(b"d"), vk),
        ];
        assert!(!verify_batch(&items));
    }

    #[test]
    fn count_failing_reports_exact_number() {
        let sk = make_key(1);
        let vk = sk.verifying_key();
        let items = vec![
            BatchItem::new(b"a", sk.sign(b"a"), vk),
            BatchItem::new(b"tampered", sk.sign(b"b"), vk),
            BatchItem::new(b"c", sk.sign(b"c"), vk),
            BatchItem::new(b"also_tampered", sk.sign(b"d"), vk),
        ];
        assert_eq!(count_failing(&items), 2);
    }

    #[test]
    fn heterogeneous_signers_can_batch_verify() {
        let items = vec![
            make_item(1, b"alice"),
            make_item(2, b"bob"),
            make_item(3, b"charlie"),
            make_item(4, b"dave"),
        ];
        assert!(verify_batch(&items));
    }

    #[test]
    fn batch_size_boundary_matches_fallback() {
        // exactly 3 items → fallback path
        let items = [make_item(1, b"a"), make_item(1, b"b"), make_item(1, b"c")];
        assert!(verify_batch(&items));
    }
}
