"""Rotation-compensated planar metric velocity, with explicit unobserved axes."""
from dataclasses import dataclass
import numpy as np
from .geometry import floor_offsets, perturb, rotation
from .tracking import CornerTracker


@dataclass
class Estimate:
    velocity_body: np.ndarray
    covariance_body: np.ndarray
    inliers: np.ndarray
    residual_rms_m: float


class MetricEstimator:
    def __init__(self, k, r_bc, t_bc, min_tracks=12, min_inlier_ratio=.6,
                 residual_threshold_m=.025, pixel_noise_px=.5, velocity_noise_mps=.02,
                 extrinsic_angle_std_rad=.005, lever_arm_std_m=.005,
                 unobserved_variance=1e6):
        self.k = np.asarray(k, float)
        self.r_bc, self.t_bc = np.asarray(r_bc, float), np.asarray(t_bc, float)
        if (self.k.shape != (3, 3) or not np.isfinite(self.k).all()
                or self.k[0, 0] <= 0 or self.k[1, 1] <= 0
                or abs(np.linalg.det(self.k)) < 1e-8):
            raise ValueError('Invalid rectified intrinsic matrix')
        if (self.r_bc.shape != (3, 3) or not np.isfinite(self.r_bc).all()
                or not np.allclose(self.r_bc.T @ self.r_bc, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(self.r_bc), 1)
                or self.t_bc.shape != (3,) or not np.isfinite(self.t_bc).all()):
            raise ValueError('Invalid measured camera extrinsics')
        if (min_tracks < 4 or not 0.5 < min_inlier_ratio <= 1
                or min(residual_threshold_m, pixel_noise_px, velocity_noise_mps,
                       extrinsic_angle_std_rad, lever_arm_std_m, unobserved_variance) <= 0):
            raise ValueError('Invalid robust estimation or uncertainty settings')
        self.min_tracks, self.min_ratio = min_tracks, min_inlier_ratio
        self.threshold, self.pixel_noise = residual_threshold_m, pixel_noise_px
        self.noise_floor = velocity_noise_mps
        self.extrinsic_angle_std, self.lever_std = extrinsic_angle_std_rad, lever_arm_std_m
        self.unobserved = unobserved_variance

    def displacements(self, p0, p1, r0, r1, h0, h1):
        a = floor_offsets(p0, self.k, r0 @ self.r_bc, h0)
        b = floor_offsets(p1, self.k, r1 @ self.r_bc, h1)
        # Camera-centre displacement minus rotation-induced camera lever-arm motion.
        return (a-b-((r1-r0) @ self.t_bc))[:, :2]

    def estimate(self, p0, p1, dt, o0, o1, a0, a1):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        if (not np.isfinite(dt) or dt <= 0 or p0.shape != p1.shape or p0.ndim != 2
                or p0.shape[1] != 2 or len(p0) < self.min_tracks
                or not np.isfinite(p0).all() or not np.isfinite(p1).all()):
            raise ValueError('Insufficient valid timestamped tracks')
        if not np.isfinite([a0.height, a1.height, a0.variance, a1.variance,
                            o0.variance, o1.variance]).all() or min(
                                a0.height, a1.height, a0.variance, a1.variance,
                                o0.variance, o1.variance) <= 0:
            raise ValueError('Scale/orientation or uncertainty is unknown')
        r0, r1 = rotation(o0.q_wb), rotation(o1.q_wb)
        d = self.displacements(p0, p1, r0, r1, a0.height, a1.height)
        # Translation RANSAC in the surveyed horizontal plane; majority support required.
        threshold = max(self.threshold, 3*self.pixel_noise*max(a0.height, a1.height)
                        / min(self.k[0, 0], self.k[1, 1]))
        candidates = d[np.random.default_rng(7).choice(len(d), min(100, len(d)), replace=False)]
        support = np.linalg.norm(d[None, :, :]-candidates[:, None, :], axis=2) < threshold
        keep = support[np.argmax(support.sum(axis=1))]
        for _ in range(3):
            centre = np.median(d[keep], axis=0)
            keep = np.linalg.norm(d-centre, axis=1) < threshold
        n = int(keep.sum())
        if n < self.min_tracks or n/len(d) < self.min_ratio:
            raise ValueError('No dominant consistent floor motion')
        if np.linalg.eigvalsh(np.cov(p0[keep].T)).min() < 25:
            raise ValueError('Features have insufficient two-dimensional spatial support')
        centre = np.mean(d[keep], axis=0)
        residuals = d[keep]-centre
        covariance = np.cov(residuals.T)/n
        # Conservative pixel-noise contribution; systematic scale/orientation errors
        # below do not shrink with the number of corners.
        covariance += np.eye(2)*(np.sqrt(2)*self.pixel_noise*max(a0.height, a1.height)
                                 / min(self.k[0, 0], self.k[1, 1]))**2/n
        base = centre
        p, q = p0[keep], p1[keep]
        eps = 1e-5
        for index, variance in ((0, a0.variance), (1, a1.variance)):
            heights = [a0.height, a1.height]
            heights[index] += eps
            shifted = self.displacements(p, q, r0, r1, *heights).mean(axis=0)
            jac = (shifted-base)/eps
            covariance += np.outer(jac, jac)*variance
        for index, variance in ((0, o0.variance), (1, o1.variance)):
            for axis in range(3):
                rs = [r0, r1]
                rs[index] = perturb(rs[index], axis, eps)
                jac = (self.displacements(p, q, *rs, a0.height, a1.height).mean(axis=0)-base)/eps
                covariance += np.outer(jac, jac)*variance
        # Measured extrinsics have finite uncertainty, shared across both observations.
        original = self.r_bc
        try:
            for axis in range(3):
                self.r_bc = perturb(original, axis, eps)
                jac = (self.displacements(p, q, r0, r1, a0.height, a1.height).mean(axis=0)-base)/eps
                covariance += np.outer(jac, jac)*self.extrinsic_angle_std**2
        finally:
            self.r_bc = original
        lever_jac = -(r1-r0)[:2, :]
        covariance += lever_jac @ lever_jac.T*self.lever_std**2
        covariance = covariance/dt**2 + np.eye(2)*self.noise_floor**2
        world_cov = np.zeros((3, 3))
        world_cov[:2, :2] = covariance
        world_cov[2, 2] = self.unobserved
        # Only horizontal translation is observed. Body-frame components retain
        # large uncertainty along the transformed plane normal.
        velocity = r1.T @ np.r_[centre/dt, 0.0]
        body_cov = r1.T @ world_cov @ r1
        return Estimate(velocity, (body_cov+body_cov.T)/2, keep,
                        float(np.sqrt(np.mean(np.sum(residuals**2, axis=1)))))


class FlowPipeline:
    def __init__(self, estimator, min_dt=.005, max_dt=.2, **tracking):
        self.estimator = estimator
        if not 0 < min_dt < max_dt:
            raise ValueError('Invalid image timestamp limits')
        self.min_dt, self.max_dt = min_dt, max_dt
        self.tracker = CornerTracker(**tracking)
        self.previous = None

    def reset(self):
        self.tracker.reset()
        self.previous = None

    def process(self, gray, stamp, orientation, altitude):
        if not np.isfinite(stamp) or stamp <= 0:
            self.reset()
            raise ValueError('Invalid image acquisition time')
        previous = self.previous
        if previous is not None:
            dt = stamp-previous[0]
            if not self.min_dt <= dt <= self.max_dt or self.tracker.previous.shape != gray.shape:
                self.reset()
                previous = None
        p0, p1 = self.tracker.update(gray)
        self.previous = (stamp, orientation, altitude)
        if previous is None:
            return None
        return self.estimator.estimate(p0, p1, stamp-previous[0], previous[1], orientation,
                                       previous[2], altitude)
