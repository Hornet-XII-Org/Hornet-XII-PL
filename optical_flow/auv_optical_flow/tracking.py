"""Shi–Tomasi features with pyramidal LK and forward/backward consistency."""
import cv2
import numpy as np


class CornerTracker:
    def __init__(self, max_corners=300, min_distance=10, fb_error_px=1.0):
        self.max_corners = max_corners
        self.min_distance = min_distance
        self.fb_error = fb_error_px
        self.reset()

    def reset(self):
        self.previous = None
        self.points = None

    def update(self, gray):
        if gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError('Expected a uint8 grayscale image')
        old, new = np.empty((0, 2)), np.empty((0, 2))
        if (self.previous is not None and self.previous.shape == gray.shape
                and self.points is not None and len(self.points) >= 4):
            opts = dict(winSize=(21, 21), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
            nxt, good, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, self.points, None, **opts)
            if nxt is not None:
                back, back_good, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, nxt, None, **opts)
                if back is not None:
                    p, q, b = self.points.reshape(-1, 2), nxt.reshape(-1, 2), back.reshape(-1, 2)
                    mask = (good.ravel().astype(bool) & back_good.ravel().astype(bool)
                            & (np.linalg.norm(p-b, axis=1) <= self.fb_error)
                            & np.isfinite(q).all(axis=1)
                            & (q[:, 0] >= 0) & (q[:, 0] < gray.shape[1])
                            & (q[:, 1] >= 0) & (q[:, 1] < gray.shape[0]))
                    old, new = p[mask].astype(float), q[mask].astype(float)
        # Retain tracks, replenish spatially separated Shi–Tomasi features.
        points = new.astype(np.float32).reshape(-1, 1, 2)
        if len(points) < self.max_corners//2:
            mask = np.full(gray.shape, 255, np.uint8)
            for p in new:
                cv2.circle(mask, tuple(np.rint(p).astype(int)), self.min_distance, 0, -1)
            fresh = cv2.goodFeaturesToTrack(gray, self.max_corners-len(points), .01,
                                           self.min_distance, mask=mask, blockSize=7)
            if fresh is not None:
                points = np.concatenate((points, fresh))
        self.points = points if len(points) else None
        self.previous = gray.copy()
        return old, new
