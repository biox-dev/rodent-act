//! Constant-velocity Kalman filter for a single track.
//!
//! State: `[cx, cy, aspect, height, vx, vy, va, vh]` (the classic SORT state
//! parameterisation). Measurement: `[cx, cy, aspect, height]`.

use nalgebra::{Matrix4, SMatrix, SVector, Vector4};

pub type Vector8 = SVector<f32, 8>;
pub type Matrix8 = SMatrix<f32, 8, 8>;

#[derive(Debug, Clone)]
pub struct KalmanState {
    pub x: Vector8,
    pub p: Matrix8,
}

pub struct KalmanFilter {
    f: Matrix8,
    h: Matrix4x8,
    q: Matrix8,
    r: Matrix4<f32>,
}

type Matrix4x8 = SMatrix<f32, 4, 8>;

impl Default for KalmanFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl KalmanFilter {
    pub fn new() -> Self {
        let mut f = Matrix8::identity();
        for i in 0..4 {
            f[(i, i + 4)] = 1.0;
        }

        let mut h = Matrix4x8::zeros();
        for i in 0..4 {
            h[(i, i)] = 1.0;
        }

        let mut q = Matrix8::identity();
        // process noise: more on the size terms, less on motion
        for i in 0..4 {
            q[(i, i)] *= 1.0;
            q[(i + 4, i + 4)] *= 0.01;
        }
        for i in 4..8 {
            q[(i, i)] = 0.01;
        }

        let mut r = Matrix4::identity();
        r[(2, 2)] *= 10.0; // aspect ratio is noisy
        r[(3, 3)] *= 10.0; // height is noisy

        Self { f, h, q, r }
    }

    pub fn initiate(&self, measurement: Vector4<f32>) -> KalmanState {
        let mut x = Vector8::zeros();
        x.fixed_rows_mut::<4>(0).copy_from(&measurement);
        let mut p = Matrix8::identity() * 10.0;
        for i in 4..8 {
            p[(i, i)] *= 100.0;
        }
        KalmanState { x, p }
    }

    pub fn predict(&self, st: &KalmanState) -> KalmanState {
        let x = self.f * st.x;
        let p = self.f * st.p * self.f.transpose() + self.q;
        KalmanState { x, p }
    }

    pub fn update(&self, st: &KalmanState, z: Vector4<f32>) -> KalmanState {
        let s = self.h * st.p * self.h.transpose() + self.r;
        let s_inv = match s.try_inverse() {
            Some(m) => m,
            None => return st.clone(),
        };
        let k = st.p * self.h.transpose() * s_inv;
        let hx: Vector4<f32> = self.h * st.x;
        let innovation: Vector4<f32> = z - hx;
        let x: Vector8 = st.x + (k * innovation).reshape_generic(nalgebra::Const::<8>, nalgebra::Const::<1>);
        let i_kh = Matrix8::identity() - k * self.h;
        // Joseph form for numerical stability
        let p = i_kh * st.p * i_kh.transpose() + k * self.r * k.transpose();
        KalmanState { x, p }
    }
}

/// `[cx, cy, aspect, height]` <-> `Bbox`.
pub fn bbox_to_z(b: &crate::common::Bbox) -> Vector4<f32> {
    let (w, h) = b.wh();
    let (cx, cy) = b.center();
    Vector4::new(cx, cy, w / h.max(1e-3), h)
}

pub fn z_to_bbox(x: &Vector8) -> crate::common::Bbox {
    let h = x[3].max(1e-3);
    let w = (x[2] * h).max(1e-3);
    crate::common::Bbox::new(x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0)
}

/// Sum of the marginal variances of the state, used as a "suspicion" score
/// (large => the track is not well supported by measurements).
pub fn state_uncertainty(st: &KalmanState) -> f32 {
    (0..8).map(|i| st.p[(i, i)]).sum::<f32>() / 8.0
}
