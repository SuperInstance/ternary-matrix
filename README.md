# ternary-matrix

Compact ternary matrix operations — 2-bit packed storage for `{-1, 0, +1}` values with multiplication, transpose, determinant, GF(3) inverse, and eigenvalue estimation.

## Why This Exists

Ternary matrices arise in quantized neural networks, ternary logic circuits, coding theory, and combinatorics. A naive matrix storing each value as a byte wastes 75% of memory on a three-valued alphabet. This crate packs 4 trits per byte (2 bits each), cutting memory usage by 4× while providing a full suite of linear algebra operations including matrix arithmetic over GF(3).

## Core Concepts

- **2-bit Packing** — Each trit stored as 2 bits: `00` = −1, `01` = 0, `10` = +1
- **TernaryMatrix** — Row-major packed matrix with get/set, arithmetic, and linear algebra
- **Ternary Multiplication** — Products clamped to `{-1, 0, +1}` or returned as full integers
- **GF(3) Inverse** — Matrix inversion over the Galois field GF(3) via Gauss-Jordan elimination
- **Power Iteration** — Dominant eigenvalue and eigenvector estimation

## Quick Start

```toml
# Cargo.toml
[dependencies]
ternary-matrix = "0.1"
```

```rust
use ternary_matrix::*;

// Create from data
let m = TernaryMatrix::from_slice(&vec![
    vec![ 1,  0, -1],
    vec![-1,  1,  0],
    vec![ 0, -1,  1],
]);

// Dimensions and access
assert_eq!(m.rows(), 3);
assert_eq!(m.get(0, 0), POS);
assert_eq!(m.get(1, 2), ZERO);

// Transpose
let t = m.transpose();
assert_eq!(t.get(0, 1), NEG);

// Matrix multiplication (ternary-clamped)
let a = TernaryMatrix::from_slice(&vec![vec![1, -1], vec![0, 1]]);
let i = TernaryMatrix::identity(2);
let product = a.multiply(&i).unwrap();

// Full integer multiplication (unclamped)
let full = a.multiply_full(&i).unwrap();

// Trace and determinant
let det = m.determinant();
let trace = m.trace();
println!("det={}, trace={}", det, trace);

// Inverse over GF(3)
let inv = m.inverse_gf3();
if let Some(inv) = inv {
    let check = m.multiply(&inv).unwrap();
    println!("A × A⁻¹ = I: {}", check.get(0, 0) == POS);
}

// Dominant eigenvalue
let (lambda, eigvec) = m.dominant_eigenvalue(100);
println!("Dominant eigenvalue: {:.4}", lambda);

// Arithmetic
let sum = a.add(&i).unwrap();
let diff = a.subtract(&i).unwrap();
let scaled = a.scale(-1);
```

## API Overview

| Method | Description |
|---|---|
| `zeros` / `identity` / `from_slice` | Constructors |
| `get` / `set` | Element access |
| `transpose` | Matrix transpose |
| `multiply` / `multiply_full` | Ternary-clamped / full-integer multiplication |
| `add` / `subtract` / `scale` | Element-wise arithmetic (clamped) |
| `trace` / `determinant` | Linear algebra fundamentals |
| `inverse_gf3` | Inversion over GF(3) |
| `dominant_eigenvalue` | Power iteration eigenvalue estimation |
| `frobenius_inner` | Frobenius inner product |
| `count_nonzero` | Sparsity metric |
| `to_vec` | Convert back to `Vec<Vec<i8>>` |

## How It Works

**Storage**: Each element occupies 2 bits within a byte. Element `(r, c)` is at byte index `(r × cols + c) / 4`, bit offset `((r × cols + c) % 4) × 2`. This gives 4× compression vs. byte-per-element storage.

**GF(3) Arithmetic**: Values are mapped to GF(3): −1 → 2, 0 → 0, +1 → 1. Inversion uses augmented matrix Gauss-Jordan elimination with modular arithmetic. Only invertible matrices (non-zero determinant mod 3) succeed.

**Eigenvalue Estimation**: Power iteration repeatedly multiplies the matrix by a vector and normalizes, converging on the dominant eigenvector. The Rayleigh quotient gives the corresponding eigenvalue.

**Determinant**: Recursive cofactor expansion (Laplace expansion). Suitable for small to moderate matrices. For large matrices, consider eigenvalue-based approaches.

## Use Cases

1. **Ternary neural network layers** — Store and compute with quantized weight matrices efficiently
2. **Coding theory** — Work with ternary codes and their generator/parity-check matrices over GF(3)
3. **Game theory** — Represent and analyze payoff matrices with ternary outcomes
4. **Combinatorial optimization** — Compact storage for large-scale ternary constraint matrices

## Ecosystem

Part of the **SuperInstance** ternary computing crate family:

- `ternary-compression-v2` — Multi-algorithm ternary compression
- `ternary-hash` — Hashing and fingerprinting for ternary data
- `ternary-pca` — Principal component analysis on ternary values
- `ternary-ga` — Genetic algorithms with ternary genomes
- `ternary-reservoir` — Echo state networks with ternary nodes
- `ternary-evolution-advanced` — Advanced evolutionary optimization
- `ternary-geometry` — Geometric algorithms in ternary space
- `ternary-causality` — Causal inference for ternary systems
- `ternary-consensus` — Distributed consensus for ternary agents

## License

MIT
