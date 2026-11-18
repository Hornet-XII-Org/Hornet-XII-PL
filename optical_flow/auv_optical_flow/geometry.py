"""Frame-explicit geometry. R_ab maps coordinates in frame b into frame a."""
from dataclasses import dataclass
import bisect
import cv2
import numpy as np


def rotation(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 0.01:
        raise ValueError('Expected a finite unit quaternion [x,y,z,w]')
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def slerp(a, b, fraction):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rotation(a)
    rotation(b)
    dot = float(a @ b)
    if dot < 0:
        b, dot = -b, -dot
    if dot > 0.9995:
        q = a + fraction * (b-a)
        return q / np.linalg.norm(q)
    theta = np.arccos(np.clip(dot, -1, 1))
    return (np.sin((1-fraction)*theta)*a + np.sin(fraction*theta)*b)/np.sin(theta)


class TimeBuffer:
    """Bounded, ordered acquisition-time buffer; never extrapolates."""
    def __init__(self, max_gap_s=0.1, capacity=2000):
        self.max_gap = max_gap_s
        self.capacity = capacity
        self.times, self.values = [], []

    def add(self, stamp, value):
        if not np.isfinite(stamp) or stamp <= 0:
            raise ValueError('Invalid acquisition timestamp')
        i = bisect.bisect_left(self.times, stamp)
        if i < len(self.times) and self.times[i] == stamp:
            self.values[i] = value
        else:
            self.times.insert(i, stamp)
            self.values.insert(i, value)
        if len(self.times) > self.capacity:
            del self.times[0]
            del self.values[0]

    def sample(self, stamp, interpolate):
        i = bisect.bisect_left(self.times, stamp)
        if i < len(self.times) and abs(self.times[i]-stamp) < 1e-8:
            return self.values[i]
        if i == 0 or i == len(self.times):
            raise ValueError('No bracketing synchronized measurements')
        a, b = self.times[i-1:i+1]
        if b-a > self.max_gap:
            raise ValueError('Supporting measurements have a timestamp gap')
        return interpolate(self.values[i-1], self.values[i], (stamp-a)/(b-a))


@dataclass
class Orientation:
    q_wb: np.ndarray
    variance: float  # conservative isotropic orientation variance, rad^2


def interpolate_orientation(a, b, f):
    return Orientation(slerp(a.q_wb, b.q_wb, f), max(a.variance, b.variance))


@dataclass
class Altitude:
    height: float  # camera height along reference +Z above a horizontal static floor
    variance: float


def interpolate_altitude(a, b, f):
    # Do not claim independent measurement noise disappears during interpolation.
    return Altitude((1-f)*a.height+f*b.height, max(a.variance, b.variance))


def camera_altitude(distance, variance, r_wb, r_br, t_br, t_bc, min_cosine=0.5):
    """Range's +X beam hits the bottom; convert slant range to camera altitude."""
    if not np.isfinite([distance, variance]).all() or distance <= 0 or variance <= 0:
        raise ValueError('Unknown/invalid range or variance')
    beam_z = (r_wb @ r_br[:, 0])[2]
    if beam_z >= -min_cosine:
        raise ValueError('Range beam is not sufficiently bottom-facing')
    h = -distance*beam_z + (r_wb @ (np.asarray(t_bc)-np.asarray(t_br)))[2]
    if h <= 0:
        raise ValueError('Camera is not above the bottom')
    return Altitude(float(h), float(variance*beam_z**2))


def floor_offsets(pixels, k, r_wc, height, min_cosine=0.2):
    """Ray intersections relative to a camera above a horizontal plane (Z=0)."""
    rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(k).T
    world = rays @ r_wc.T
    norm = np.linalg.norm(world, axis=1)
    if np.any(world[:, 2] / norm >= -min_cosine):
        raise ValueError('Camera rays do not safely intersect the bottom plane')
    return world * (-height/world[:, 2])[:, None]


def perturb(r, axis, epsilon):
    v = np.zeros(3)
    v[axis] = epsilon
    return cv2.Rodrigues(v)[0] @ r
