//! Hungarian assignment (shortest augmenting path, Munkres-Jonker-Volgenant
//! flavour, `O(n^3)`). Cost matrix may be rectangular.

/// Returns `assignment[i] = j` for each row `i` (or `None` if unmatched).
/// `cost` is interpreted as a *minimisation* problem.
pub fn assign(cost: &[Vec<f32>]) -> Vec<Option<usize>> {
    let n = cost.len();
    if n == 0 {
        return Vec::new();
    }
    // Pad to square with a large dummy cost
    let m = cost[0].len();
    let size = n.max(m);
    let mut c = vec![vec![1e6f32; size]; size];
    for i in 0..n {
        for j in 0..m {
            c[i][j] = cost[i][j];
        }
    }

    // Core algorithm (1-indexed internally), adapted from scipy's
    // linear_sum_assignment.
    let mut u = vec![0.0f32; size + 1];
    let mut v = vec![0.0f32; size + 1];
    let mut p = vec![0usize; size + 1];
    let mut way = vec![0usize; size + 1];
    for i in 1..=size {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![f32::INFINITY; size + 1];
        let mut used = vec![false; size + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f32::INFINITY;
            let mut j1 = 0usize;
            for j in 1..=size {
                if used[j] {
                    continue;
                }
                let cur = c[i0 - 1][j - 1] - u[i0] - v[j];
                if cur < minv[j] {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if minv[j] < delta {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for j in 0..=size {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut result = vec![None; n];
    for j in 1..=size {
        if p[j] == 0 {
            continue;
        }
        let i = p[j] - 1;
        let cij = if i < n && j - 1 < m { cost[i][j - 1] } else { f32::INFINITY };
        if i < n && j - 1 < m && cij < 1e5 {
            result[i] = Some(j - 1);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple() {
        let c = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        assert_eq!(assign(&c), vec![Some(0), Some(1)]);
    }

    #[test]
    fn forced_swap() {
        let c = vec![vec![5.0, 1.0], vec![1.0, 5.0]];
        assert_eq!(assign(&c), vec![Some(1), Some(0)]);
    }

    #[test]
    fn rectangular() {
        let c = vec![vec![1.0, 9.0, 9.0], vec![9.0, 9.0, 2.0]];
        assert_eq!(assign(&c), vec![Some(0), Some(2)]);
    }

    #[test]
    fn gated_out() {
        let c = vec![vec![1e6, 1e6]];
        assert_eq!(assign(&c), vec![None]);
    }
}
