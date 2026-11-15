"""Ball detection and near-axis size-based position estimation."""

import math

import cv2

from geometry_helpers import _search_roi, _camera_matrix, _parameters_assumed


def get_ball_position(image, bbox, *, min_area=20, min_circularity=0.75,
                      canny_low=50, canny_high=150, blur_kernel=5,
                      camera=None, diameter_m=None, diameter_assumed=True,
                      geometry=None):
    """Find a circular candidate; metric output uses the near-axis approximation."""
    if (not math.isfinite(min_area) or min_area <= 0 or not 0 <= min_circularity <= 1
            or not 0 <= canny_low < canny_high <= 255 or not isinstance(blur_kernel, int)
            or blur_kernel < 1 or blur_kernel % 2 == 0):
        raise ValueError("Check area, circularity, Canny thresholds and the positive odd blur kernel")
    roi, (x1, y1, x2, y2) = _search_roi(image, bbox)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    edges = cv2.Canny(gray, canny_low, canny_high)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    roi_height, roi_width = roi.shape[:2]
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area < min_area or perimeter == 0:
            continue
        circularity = 4 * math.pi * area / perimeter**2
        if circularity < min_circularity:
            continue
        bx, by, bw, bh = cv2.boundingRect(contour)
        if bx <= 0 or by <= 0 or bx + bw >= roi_width or by + bh >= roi_height:
            continue
        (u, v), radius = cv2.minEnclosingCircle(contour)
        candidates.append((area, u, v, radius))

    if not candidates:
        return {"status": "no_circle_found", "center_px": None,
                "radius_px": None, "xyz_m": None,"search_bbox_px":[x1,y1,x2,y2]}

    _, u, v, radius = max(candidates, key=lambda item: item[0])
    center = (u + x1, v + y1)
    result = {"status": "candidate_needs_review", "center_px": center,
              "radius_px": radius, "xyz_m": None, "distance_m": None,
              "search_bbox_px":[x1,y1,x2,y2]}
    if camera is not None and diameter_m is not None:
        result.update(_ball_geometry(center, radius, image.shape, camera,
                                     diameter_m, diameter_assumed, geometry or {}))
    else:
        result["geometry_status"] = "missing_camera_or_diameter"
    return result


def _ball_geometry(center, radius, image_shape, camera, diameter_m,
                   diameter_assumed, geometry):
    K = _camera_matrix(camera, image_shape)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if not all(math.isfinite(v) for v in (radius, diameter_m, *center)) or min(radius, diameter_m) <= 0:
        raise ValueError("Geometry inputs must be finite; radius and diameter must be positive")
    assumed = _parameters_assumed(camera, diameter_assumed)
    a, b = (center[0] - cx) / fx, (center[1] - cy) / fy
    norm = math.sqrt(a*a + b*b + 1)
    result = {"frame": "camera_optical", "target": "ball_center",
              "ray_unit": [a / norm, b / norm, 1 / norm],
              "parameters_assumed": bool(assumed),
              "geometry_method": "near_axis_pinhole_size_approximation"}
    if math.hypot(a, b) > geometry.get("max_normalized_offset", 0.35):
        return {**result, "geometry_status": "outside_near_axis_approximation"}
    if abs(fx - fy) / max(fx, fy) > geometry.get("max_focal_relative_difference", 0.01):
        return {**result, "geometry_status": "unequal_focal_lengths_not_supported"}
    z = ((fx + fy) / 2) * diameter_m / (2 * radius)
    xyz = [a * z, b * z, z]
    return {**result, "xyz_m": xyz, "distance_m": z * norm,
            "geometry_status": "approximate_assumed" if assumed else "approximate"}
