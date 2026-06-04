# Future Integration: ternary-matrix

## Current State
Implements compact matrix operations for ternary values: `TernaryMatrix` with 2-bit-per-element packing (4 elements per byte), matrix multiplication, transpose, determinant, inverse over GF(3), trace, eigenvalue estimation, and Kronecker product.

## Integration Opportunities

### With ternary-tensor
`TernaryMatrix` is the foundation for `ternary-tensor`. A rank-3 tensor is a vector of `TernaryMatrix` slices. The 2-bit packing extends naturally: a rank-3 tensor of shape (d, h, w) uses d × h × w × 2 bits. The `multiply()` method becomes tensor contraction. This crate provides the storage layer that `ternary-tensor` builds upon.

### With ternary-cell
The cell grid IS a `TernaryMatrix`. Each cell's state is a trit; the grid is a matrix. Grid operations (neighborhood extraction, bulk updates, pattern matching) are matrix operations. The 2-bit packing means a 64×64 cell grid uses only 1KB — fits in ESP32 SRAM with room to spare. `multiply()` computes grid convolution (weighted neighbor sums).

### With ternary-pca / ternary-projection
Both crates need matrix operations (covariance computation, eigenvalue decomposition). Currently they implement their own. `TernaryMatrix` should be the shared linear algebra backend, with `ternary-pca` calling `TernaryMatrix::multiply()` and `TernaryMatrix::transpose()` instead of re-implementing them.

## Potential in Mature Systems
In PLATO, `TernaryMatrix` is the universal data structure for Layer 0. Every grid, every weight matrix, every lookup table is a `TernaryMatrix`. The 2-bit packing means memory bandwidth is 4× better than byte-per-element. `gf3_inverse()` enables solving linear systems over GF(3) — the basis for ternary error-correcting codes in `ternary-codes`. On ESP32, the packed format is a natural fit for DMA transfers.

## Cross-Pollination Ideas
**Cryptography × Matrix:** GF(3) matrix operations are the foundation for ternary lattice cryptography. `gf3_inverse()` + `determinant()` enable key generation. `kronecker_product()` constructs higher-dimensional lattices from smaller building blocks. This could be the basis for post-quantum ternary crypto.

**Games × Matrix:** Board games on ternary grids (tic-tac-toe generalizations) are matrix games. `multiply()` computes position evaluations. `transpose()` checks symmetry. `determinant()` over GF(3) detects winning positions in generalized tic-tac-toe. Connects to `ternary-games`.

## Dependencies for Next Steps
- Promote to foundational crate: `ternary-tensor`, `ternary-pca`, `ternary-projection` should depend on this
- SIMD acceleration for packed multiply on ARM (ESP32-S3 has SIMD)
- Sparse matrix variant for large grids with mostly-zero cells
