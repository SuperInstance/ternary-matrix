#![forbid(unsafe_code)]

//! Matrix operations optimized for ternary values ({-1, 0, +1}).
//!
//! Uses compact storage: 2 bits per trit. Supports matrix multiplication,
//! transpose, determinant, inverse over GF(3), trace, and eigenvalue estimation.

/// A trit value.
pub type Trit = i8;

pub const NEG: Trit = -1;
pub const ZERO: Trit = 0;
pub const POS: Trit = 1;

/// Trit stored as 2 bits: -1 -> 0b00, 0 -> 0b01, +1 -> 0b10
fn trit_to_bits(t: Trit) -> u8 {
    match t {
        -1 => 0b00,
        0 => 0b01,
        1 => 0b10,
        _ => 0b01,
    }
}

fn bits_to_trit(b: u8) -> Trit {
    match b & 0b11 {
        0b00 => NEG,
        0b01 => ZERO,
        0b10 => POS,
        _ => ZERO,
    }
}

/// Compact ternary matrix: 2 bits per element, 4 elements per byte.
#[derive(Clone, Debug)]
pub struct TernaryMatrix {
    rows: usize,
    cols: usize,
    /// Packed storage: each byte holds 4 trits (2 bits each).
    /// Element (r, c) is at byte index (r*cols + c) / 4, bits ((r*cols + c) % 4) * 2.
    data: Vec<u8>,
}

impl TernaryMatrix {
    /// Create a zero matrix of given dimensions.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        let total = rows * cols;
        let bytes = (total + 3) / 4;
        // Initialize with all zeros (trit 0 = bits 01)
        let mut data = vec![0u8; bytes];
        for i in 0..total {
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            data[byte_idx] |= trit_to_bits(ZERO) << bit_offset;
        }
        Self { rows, cols, data }
    }

    /// Create matrix from a 2D slice of trits.
    pub fn from_slice(data: &[Vec<Trit>]) -> Self {
        if data.is_empty() {
            return Self::zeros(0, 0);
        }
        let rows = data.len();
        let cols = data[0].len();
        let mut m = Self::zeros(rows, cols);
        for (r, row) in data.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                m.set(r, c, v);
            }
        }
        m
    }

    /// Create an identity matrix of size n.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.set(i, i, POS);
        }
        m
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get element at (r, c).
    pub fn get(&self, r: usize, c: usize) -> Trit {
        let idx = r * self.cols + c;
        let byte_idx = idx / 4;
        let bit_offset = (idx % 4) * 2;
        bits_to_trit((self.data[byte_idx] >> bit_offset) & 0b11)
    }

    /// Set element at (r, c).
    pub fn set(&mut self, r: usize, c: usize, value: Trit) {
        let idx = r * self.cols + c;
        let byte_idx = idx / 4;
        let bit_offset = (idx % 4) * 2;
        self.data[byte_idx] &= !(0b11 << bit_offset); // clear
        self.data[byte_idx] |= trit_to_bits(value) << bit_offset; // set
    }

    /// Convert to a 2D vector of trits.
    pub fn to_vec(&self) -> Vec<Vec<Trit>> {
        (0..self.rows)
            .map(|r| (0..self.cols).map(|c| self.get(r, c)).collect())
            .collect()
    }

    /// Matrix transpose.
    pub fn transpose(&self) -> TernaryMatrix {
        let mut result = TernaryMatrix::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                result.set(c, r, self.get(r, c));
            }
        }
        result
    }

    /// Matrix multiplication. Returns None if dimensions don't match.
    pub fn multiply(&self, other: &TernaryMatrix) -> Option<TernaryMatrix> {
        if self.cols != other.rows {
            return None;
        }
        let mut result = TernaryMatrix::zeros(self.rows, other.cols);
        for r in 0..self.rows {
            for c in 0..other.cols {
                let mut sum: i32 = 0;
                for k in 0..self.cols {
                    sum += self.get(r, k) as i32 * other.get(k, c) as i32;
                }
                // Clamp to ternary range for ternary multiplication result
                // Actually, keep full value for utility — but offer ternary_clamp
                result.set(r, c, sum.clamp(-1, 1) as Trit);
            }
        }
        Some(result)
    }

    /// Matrix multiplication returning full integer values (not clamped to ternary).
    pub fn multiply_full(&self, other: &TernaryMatrix) -> Option<Vec<Vec<i32>>> {
        if self.cols != other.rows {
            return None;
        }
        let mut result = vec![vec![0i32; other.cols]; self.rows];
        for r in 0..self.rows {
            for c in 0..other.cols {
                let mut sum: i32 = 0;
                for k in 0..self.cols {
                    sum += self.get(r, k) as i32 * other.get(k, c) as i32;
                }
                result[r][c] = sum;
            }
        }
        Some(result)
    }

    /// Compute the trace (sum of diagonal).
    pub fn trace(&self) -> i32 {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.get(i, i) as i32).sum()
    }

    /// Compute determinant recursively. Works for small matrices.
    pub fn determinant(&self) -> i32 {
        if self.rows != self.cols {
            return 0;
        }
        let n = self.rows;
        match n {
            0 => 1,
            1 => self.get(0, 0) as i32,
            2 => {
                self.get(0, 0) as i32 * self.get(1, 1) as i32
                    - self.get(0, 1) as i32 * self.get(1, 0) as i32
            }
            _ => {
                let mut det = 0i32;
                for j in 0..n {
                    let sub = self.submatrix(0, j);
                    let sign = if j % 2 == 0 { 1 } else { -1 };
                    det += sign * self.get(0, j) as i32 * sub.determinant();
                }
                det
            }
        }
    }

    /// Get submatrix with given row and column removed.
    fn submatrix(&self, skip_row: usize, skip_col: usize) -> TernaryMatrix {
        let mut result = TernaryMatrix::zeros(self.rows - 1, self.cols - 1);
        let mut rr = 0;
        for r in 0..self.rows {
            if r == skip_row { continue; }
            let mut cc = 0;
            for c in 0..self.cols {
                if c == skip_col { continue; }
                result.set(rr, cc, self.get(r, c));
                cc += 1;
            }
            rr += 1;
        }
        result
    }

    /// Check if square.
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// Count non-zero elements.
    pub fn count_nonzero(&self) -> usize {
        let mut count = 0;
        for r in 0..self.rows {
            for c in 0..self.cols {
                if self.get(r, c) != ZERO {
                    count += 1;
                }
            }
        }
        count
    }

    /// Frobenius inner product with another matrix.
    pub fn frobenius_inner(&self, other: &TernaryMatrix) -> i32 {
        if self.rows != other.rows || self.cols != other.cols {
            return 0;
        }
        let mut sum = 0i32;
        for r in 0..self.rows {
            for c in 0..self.cols {
                sum += self.get(r, c) as i32 * other.get(r, c) as i32;
            }
        }
        sum
    }

    /// Scale by a constant (clamped to ternary).
    pub fn scale(&self, factor: i32) -> TernaryMatrix {
        let mut result = TernaryMatrix::zeros(self.rows, self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                let v = (self.get(r, c) as i32 * factor).clamp(-1, 1);
                result.set(r, c, v as Trit);
            }
        }
        result
    }

    /// Element-wise addition (clamped to ternary).
    pub fn add(&self, other: &TernaryMatrix) -> Option<TernaryMatrix> {
        if self.rows != other.rows || self.cols != other.cols {
            return None;
        }
        let mut result = TernaryMatrix::zeros(self.rows, self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                let v = (self.get(r, c) as i32 + other.get(r, c) as i32).clamp(-1, 1);
                result.set(r, c, v as Trit);
            }
        }
        Some(result)
    }

    /// Element-wise subtraction (clamped to ternary).
    pub fn subtract(&self, other: &TernaryMatrix) -> Option<TernaryMatrix> {
        if self.rows != other.rows || self.cols != other.cols {
            return None;
        }
        let mut result = TernaryMatrix::zeros(self.rows, self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                let v = (self.get(r, c) as i32 - other.get(r, c) as i32).clamp(-1, 1);
                result.set(r, c, v as Trit);
            }
        }
        Some(result)
    }

    /// Compute inverse over GF(3) for square matrices.
    /// Returns None if the matrix is singular.
    /// In GF(3): values are {0, 1, 2} mapping to {-1→2, 0→0, 1→1}.
    pub fn inverse_gf3(&self) -> Option<TernaryMatrix> {
        if !self.is_square() || self.rows == 0 {
            return None;
        }
        let n = self.rows;
        // Augmented matrix [A | I]
        let mut aug = vec![vec![0i32; 2 * n]; n];
        for r in 0..n {
            for c in 0..n {
                aug[r][c] = trit_to_gf3(self.get(r, c));
                aug[r][c + n] = if r == c { 1 } else { 0 };
            }
        }

        // Gauss-Jordan elimination over GF(3)
        for col in 0..n {
            // Find pivot
            let mut pivot_row = None;
            for r in col..n {
                if aug[r][col] != 0 {
                    pivot_row = Some(r);
                    break;
                }
            }
            let pivot_row = pivot_row?;
            // Swap rows
            aug.swap(col, pivot_row);
            // Scale pivot row so pivot = 1
            let pivot_val = aug[col][col];
            let inv = gf3_inverse(pivot_val)?;
            for j in 0..2 * n {
                aug[col][j] = (aug[col][j] * inv) % 3;
            }
            // Eliminate column
            for r in 0..n {
                if r != col && aug[r][col] != 0 {
                    let factor = aug[r][col];
                    for j in 0..2 * n {
                        aug[r][j] = (aug[r][j] - factor * aug[col][j] + 9) % 3;
                    }
                }
            }
        }

        // Extract inverse
        let mut result = TernaryMatrix::zeros(n, n);
        for r in 0..n {
            for c in 0..n {
                result.set(r, c, gf3_to_trit(aug[r][c + n]));
            }
        }
        Some(result)
    }

    /// Estimate eigenvalues using power iteration (dominant eigenvalue only).
    /// Returns (eigenvalue, eigenvector) as f64 values.
    pub fn dominant_eigenvalue(&self, iterations: usize) -> (f64, Vec<f64>) {
        let n = self.rows.min(self.cols);
        if n == 0 {
            return (0.0, vec![]);
        }
        // Start with uniform vector
        let mut v = vec![1.0f64; n];
        let norm = (n as f64).sqrt();
        for x in &mut v { *x /= norm; }

        for _ in 0..iterations {
            let mut w = vec![0.0f64; n];
            for i in 0..n {
                for j in 0..n {
                    w[i] += self.get(i, j) as f64 * v[j];
                }
            }
            let norm_w: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm_w > 1e-12 {
                for x in &mut w { *x /= norm_w; }
            }
            v = w;
        }
        // Rayleigh quotient
        let mut mv = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                mv[i] += self.get(i, j) as f64 * v[j];
            }
        }
        let lambda: f64 = v.iter().zip(mv.iter()).map(|(a, b)| a * b).sum();
        (lambda, v)
    }
}

/// Convert trit to GF(3) element: -1 -> 2, 0 -> 0, +1 -> 1.
fn trit_to_gf3(t: Trit) -> i32 {
    match t {
        -1 => 2,
        0 => 0,
        1 => 1,
        _ => 0,
    }
}

/// Convert GF(3) element to trit: 0 -> 0, 1 -> +1, 2 -> -1.
fn gf3_to_trit(v: i32) -> Trit {
    match v % 3 {
        0 => ZERO,
        1 => POS,
        2 => NEG,
        _ => ZERO,
    }
}

/// Multiplicative inverse in GF(3): 1 -> 1, 2 -> 2, 0 -> None.
fn gf3_inverse(v: i32) -> Option<i32> {
    match v % 3 {
        1 => Some(1),
        2 => Some(2),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros_matrix() {
        let m = TernaryMatrix::zeros(3, 4);
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 4);
        for r in 0..3 {
            for c in 0..4 {
                assert_eq!(m.get(r, c), ZERO);
            }
        }
    }

    #[test]
    fn test_identity() {
        let m = TernaryMatrix::identity(3);
        for r in 0..3 {
            for c in 0..3 {
                if r == c {
                    assert_eq!(m.get(r, c), POS);
                } else {
                    assert_eq!(m.get(r, c), ZERO);
                }
            }
        }
    }

    #[test]
    fn test_from_slice() {
        let data = vec![vec![1, 0, -1], vec![-1, 1, 0]];
        let m = TernaryMatrix::from_slice(&data);
        assert_eq!(m.get(0, 0), POS);
        assert_eq!(m.get(0, 1), ZERO);
        assert_eq!(m.get(0, 2), NEG);
        assert_eq!(m.get(1, 0), NEG);
        assert_eq!(m.get(1, 1), POS);
    }

    #[test]
    fn test_set_and_get() {
        let mut m = TernaryMatrix::zeros(2, 2);
        m.set(0, 0, POS);
        m.set(0, 1, NEG);
        m.set(1, 0, ZERO);
        m.set(1, 1, POS);
        assert_eq!(m.get(0, 0), POS);
        assert_eq!(m.get(0, 1), NEG);
        assert_eq!(m.get(1, 0), ZERO);
        assert_eq!(m.get(1, 1), POS);
    }

    #[test]
    fn test_transpose() {
        let data = vec![vec![1, 0], vec![-1, 1], vec![0, -1]];
        let m = TernaryMatrix::from_slice(&data);
        let t = m.transpose();
        assert_eq!(t.rows(), 2);
        assert_eq!(t.cols(), 3);
        assert_eq!(t.get(0, 0), POS);
        assert_eq!(t.get(1, 0), ZERO);
        assert_eq!(t.get(0, 1), NEG);
        assert_eq!(t.get(1, 1), POS);
    }

    #[test]
    fn test_multiply_identity() {
        let a = TernaryMatrix::from_slice(&vec![vec![1, -1], vec![0, 1]]);
        let i = TernaryMatrix::identity(2);
        let result = a.multiply(&i).unwrap();
        assert_eq!(result.get(0, 0), POS);
        assert_eq!(result.get(0, 1), NEG);
        assert_eq!(result.get(1, 0), ZERO);
        assert_eq!(result.get(1, 1), POS);
    }

    #[test]
    fn test_multiply_incompatible() {
        let a = TernaryMatrix::zeros(2, 3);
        let b = TernaryMatrix::zeros(2, 2);
        assert!(a.multiply(&b).is_none());
    }

    #[test]
    fn test_multiply_full() {
        let a = TernaryMatrix::from_slice(&vec![vec![1, 1], vec![1, 1]]);
        let b = TernaryMatrix::from_slice(&vec![vec![1, 0], vec![0, 1]]);
        let result = a.multiply_full(&b).unwrap();
        assert_eq!(result[0][0], 1);
        assert_eq!(result[0][1], 1);
    }

    #[test]
    fn test_trace() {
        let m = TernaryMatrix::from_slice(&vec![vec![1, 0, -1], vec![0, 1, 0], vec![-1, 0, 1]]);
        assert_eq!(m.trace(), 3);
    }

    #[test]
    fn test_determinant_2x2() {
        let m = TernaryMatrix::from_slice(&vec![vec![1, 0], vec![0, 1]]);
        assert_eq!(m.determinant(), 1);
    }

    #[test]
    fn test_determinant_3x3() {
        let m = TernaryMatrix::from_slice(&vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]);
        assert_eq!(m.determinant(), 1);
    }

    #[test]
    fn test_determinant_singular() {
        let m = TernaryMatrix::from_slice(&vec![vec![1, 1], vec![1, 1]]);
        assert_eq!(m.determinant(), 0);
    }

    #[test]
    fn test_count_nonzero() {
        let m = TernaryMatrix::from_slice(&vec![vec![1, 0, -1], vec![0, 0, 0]]);
        assert_eq!(m.count_nonzero(), 2);
    }

    #[test]
    fn test_frobenius_inner() {
        let a = TernaryMatrix::from_slice(&vec![vec![1, 0], vec![-1, 1]]);
        let b = TernaryMatrix::from_slice(&vec![vec![1, 0], vec![-1, 1]]);
        // 1*1 + 0*0 + (-1)*(-1) + 1*1 = 3
        assert_eq!(a.frobenius_inner(&b), 3);
    }

    #[test]
    fn test_add() {
        let a = TernaryMatrix::from_slice(&vec![vec![1, 0], vec![0, -1]]);
        let b = TernaryMatrix::from_slice(&vec![vec![0, 1], vec![1, 0]]);
        let c = a.add(&b).unwrap();
        assert_eq!(c.get(0, 0), POS); // 1+0 = 1
        assert_eq!(c.get(0, 1), POS); // 0+1 = 1
        assert_eq!(c.get(1, 0), POS); // 0+1 = 1
        // -1+0 = -1
        assert_eq!(c.get(1, 1), NEG);
    }

    #[test]
    fn test_subtract() {
        let a = TernaryMatrix::from_slice(&vec![vec![1, 0]]);
        let b = TernaryMatrix::from_slice(&vec![vec![0, 1]]);
        let c = a.subtract(&b).unwrap();
        assert_eq!(c.get(0, 0), POS); // 1-0 = 1
        assert_eq!(c.get(0, 1), NEG); // 0-1 = -1
    }

    #[test]
    fn test_scale() {
        let m = TernaryMatrix::from_slice(&vec![vec![1, 0, -1]]);
        let s = m.scale(-1);
        assert_eq!(s.get(0, 0), NEG);
        assert_eq!(s.get(0, 1), ZERO);
        assert_eq!(s.get(0, 2), POS);
    }

    #[test]
    fn test_inverse_gf3_identity() {
        let i = TernaryMatrix::identity(2);
        let inv = i.inverse_gf3().unwrap();
        // Inverse of identity is identity
        assert_eq!(inv.get(0, 0), POS);
        assert_eq!(inv.get(1, 1), POS);
        assert_eq!(inv.get(0, 1), ZERO);
    }

    #[test]
    fn test_inverse_gf3_singular() {
        let m = TernaryMatrix::from_slice(&vec![vec![1, 1], vec![1, 1]]);
        assert!(m.inverse_gf3().is_none());
    }

    #[test]
    fn test_inverse_gf3_roundtrip() {
        // A matrix invertible over GF(3)
        let m = TernaryMatrix::from_slice(&vec![vec![1, 0], vec![0, 1]]);
        let inv = m.inverse_gf3().unwrap();
        let product = m.multiply(&inv).unwrap();
        // Should be identity
        assert_eq!(product.get(0, 0), POS);
        assert_eq!(product.get(0, 1), ZERO);
        assert_eq!(product.get(1, 0), ZERO);
        assert_eq!(product.get(1, 1), POS);
    }

    #[test]
    fn test_dominant_eigenvalue() {
        let m = TernaryMatrix::identity(2);
        let (lambda, v) = m.dominant_eigenvalue(100);
        assert!((lambda - 1.0).abs() < 0.1, "eigenvalue should be ~1.0, got {}", lambda);
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_to_vec_roundtrip() {
        let data = vec![vec![1, 0, -1], vec![-1, 1, 0]];
        let m = TernaryMatrix::from_slice(&data);
        let out = m.to_vec();
        assert_eq!(out, data);
    }

    #[test]
    fn test_compact_storage_efficiency() {
        let m = TernaryMatrix::zeros(4, 4); // 16 elements = 4 bytes
        assert_eq!(m.data.len(), 4);
    }
}
