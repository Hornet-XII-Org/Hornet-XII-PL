"""Shared camera/ellipse geometry and image measurements used by the object modules."""

import math

import cv2
import numpy as np


def _crop_roi(image, bbox, min_size=1):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3):
        raise ValueError("image must be a uint8 BGR image")
    x1, y1, x2, y2 = map(int, bbox)
    if not (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]):
        raise ValueError("bbox must lie inside the image")
    if min(x2-x1, y2-y1) < min_size:
        raise ValueError(f"ROI must be at least {min_size} by {min_size} pixels")
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def _search_roi(image, bbox, min_size=1):
    """Allow small detector-box errors using real image pixels around the box."""
    _, (x1,y1,x2,y2) = _crop_roi(image,bbox,min_size)
    margin = max(6,math.ceil(.12*min(x2-x1,y2-y1)))
    bounds = (max(0,x1-margin),max(0,y1-margin),
              min(image.shape[1],x2+margin),min(image.shape[0],y2+margin))
    return _crop_roi(image,bounds,min_size)


def _parameters_assumed(camera, diameter_assumed):
    return bool(camera.get("assumed", True) or diameter_assumed
                or not camera.get("input_rectified", False)
                or any(camera.get(k) is None for k in ("cx_px", "cy_px", "image_width", "image_height")))


def camera_for_image(camera, image_shape):
    """Unknown photos use an explicit FOV assumption, never another camera's K."""
    camera = dict(camera or {"mode": "estimate", "horizontal_fov_deg": 60.0})
    if camera.get("mode", "intrinsics") == "estimate":
        fov = float(camera.get("horizontal_fov_deg", 60.0))
        if not math.isfinite(fov) or not 5 < fov < 175:
            raise ValueError("horizontal_fov_deg must be between 5 and 175 degrees")
        height, width = image_shape[:2]
        focal = width / (2*math.tan(math.radians(fov)/2))
        camera.update(fx_px=focal, fy_px=focal, cx_px=width/2, cy_px=height/2,
                      image_width=width, image_height=height, assumed=True,
                      input_rectified=False, horizontal_fov_deg=fov)
    elif camera.get("mode", "intrinsics") != "intrinsics":
        raise ValueError("camera.mode must be estimate or intrinsics")
    _camera_matrix(camera, image_shape)
    return camera


def _camera_matrix(camera, image_shape):
    height, width = image_shape[:2]
    fx, fy = float(camera["fx_px"]), float(camera["fy_px"])
    cx = width / 2 if camera.get("cx_px") is None else float(camera["cx_px"])
    cy = height / 2 if camera.get("cy_px") is None else float(camera["cy_px"])
    if not all(math.isfinite(v) for v in (fx, fy, cx, cy)) or min(fx, fy) <= 0:
        raise ValueError("Camera parameters must be finite; focal lengths must be positive")
    for key, actual in (("image_width", width), ("image_height", height)):
        if camera.get(key) is not None and camera[key] != actual:
            raise ValueError(f"{key} does not match the image; update calibration")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)


def _rotation(angle):
    a = math.radians(angle)
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def _ellipse_points(ellipse, count=120):
    center, axes, angle = ellipse
    t = np.linspace(0, 2 * np.pi, count, endpoint=False)
    return (np.column_stack((np.cos(t), np.sin(t))) * (np.array(axes) / 2)) @ _rotation(angle).T + center


def _ellipse_conic(ellipse):
    center, axes, angle = ellipse
    rotation = _rotation(angle)
    q = rotation @ np.diag(1 / (np.array(axes) / 2)**2) @ rotation.T
    center = np.array(center)
    return np.block([[q, (-q @ center)[:, None]],
                      [(-center @ q)[None, :], np.array([[center @ q @ center - 1]])]])


def _circle_poses_from_ellipse(ellipse, K, radius_m):
    """Two circular sections of the viewing cone; normals point into the drum."""
    cone = K.T @ _ellipse_conic(ellipse) @ K
    cone /= np.linalg.norm(cone)
    values, vectors = np.linalg.eigh(cone)
    if values[1] < 0:
        cone = -cone
        values, vectors = np.linalg.eigh(cone)
    low, middle, high = values
    if not low < 0 < middle:
        raise ValueError("Invalid viewing cone")
    tilt = max(0, (high-middle) / (high-low))
    if tilt < 1e-7:  # Float32 ellipse output can split an exactly frontal circle.
        tilt = 0.
    nx, nz = math.sqrt(tilt), math.sqrt(max(0, 1-tilt))
    poses = []
    for sign in (-1, 1):
        normal = sign * nx * vectors[:, 2] + nz * vectors[:, 0]
        if normal[2] < 0:
            normal = -normal
        offset = -(cone @ normal - normal * (normal @ cone @ normal)) / middle
        radius0_sq = offset @ offset - (normal @ cone @ normal) / middle
        if radius0_sq <= 0:
            continue
        xyz = (normal + offset) * radius_m / math.sqrt(radius0_sq)
        if xyz[2] <= 0:
            normal, xyz = -normal, -xyz
        if any(np.linalg.norm(xyz - p["xyz_m"]) < 1e-6
               and abs(normal @ p["normal_into_drum"]) > 1-1e-8 for p in poses):
            continue
        pixel = K @ xyz
        poses.append({"id": chr(65 + len(poses)), "xyz_m": xyz.tolist(),
                      "normal_into_drum": normal.tolist(),
                      "center_px": (pixel[:2] / pixel[2]).tolist()})
    return poses


def _fit_ellipse(points):
    """OpenCV fit with normalized coordinates and a consistent major-axis angle."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) < 8 or not np.isfinite(points).all():
        raise ValueError("Insufficient finite rim points")
    mean = points.mean(axis=0)
    scale = np.sqrt(np.mean(np.sum((points-mean)**2, axis=1)))
    if scale < 1e-6 or np.linalg.matrix_rank(points-mean) < 2:
        raise ValueError("Degenerate rim points")
    try:
        center, axes, angle = cv2.fitEllipseDirect(((points-mean)/scale).astype(np.float32))
    except cv2.error as exc:
        raise ValueError("Ellipse fit failed") from exc
    if not np.isfinite([*center, *axes, angle]).all() or min(axes) <= 0:
        raise ValueError("Invalid ellipse dimensions")
    if axes[0] < axes[1]:
        axes, angle = axes[::-1], angle+90
    return [(np.array(center)*scale+mean).tolist(),
            (np.array(axes)*scale).tolist(), angle % 180]


def _local_ellipse(points, ellipse):
    center, axes, angle = ellipse
    return (points-center) @ _rotation(angle) / (np.array(axes)/2)


def _mask(image, bbox, object_mask):
    if object_mask is None:
        return None
    mask = np.asarray(object_mask)
    if mask.shape != image.shape[:2] or not np.isfinite(mask).all() or not np.isin(mask, [0, 1, 255]).all():
        raise ValueError("object_mask must be a full-image binary 0/1 or 0/255 mask")
    x1, y1, x2, y2 = bbox
    return (mask[y1:y2, x1:x2] != 0).astype(np.uint8)*255


def _color_mask(roi, color):
    ranges = {"red": [(0, 12), (168, 179)], "orange": [(5, 25)],
              "yellow": [(20, 40)], "blue": [(85, 135)], "green": [(35, 90)]}
    if color not in ranges:
        raise ValueError("Set a supported color, or supply an object_mask")
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv.transpose(2, 0, 1)
    selected = np.zeros(hue.shape, bool)
    for low, high in ranges[color]:
        selected |= (hue >= low) & (hue <= high)
    return (selected & (saturation >= 70) & (value >= 35)).astype(np.uint8)*255


def _side_fit(t, s):
    first, second = np.triu_indices(len(t), 1)
    slope = float(np.median((s[second]-s[first])/(t[second]-t[first])))
    fit = np.array([slope, np.median(s-slope*t)])
    keep = np.ones(len(t), bool)
    for _ in range(3):
        residual = np.abs(s-(fit[0]*t+fit[1]))
        updated = residual <= max(.75, 2.5*float(np.median(residual)))
        if updated.sum() < 4:
            break
        keep = updated
        fit = np.linalg.lstsq(np.column_stack((t[keep], np.ones(keep.sum()))), s[keep], rcond=None)[0]
    residual = np.abs(s-(fit[0]*t+fit[1]))
    return fit, float(np.sqrt(np.mean(residual[keep]**2)))


def _shaft(mask, offset, combine=False):
    """Measure both sides perpendicular to the shaft, trimming its end caps."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    groups = [c.reshape(-1, 2).astype(float) for c in contours if cv2.contourArea(c) >= 8]
    if combine and groups:
        groups = [np.vstack(groups)]  # Gate markings can be disconnected stripes.
    best = None
    for points in groups:
        if len(points) < 20:
            continue
        center = points.mean(axis=0)
        _, _, vectors = np.linalg.svd(points-center, full_matrices=False)
        axis = vectors[0]
        if axis[1] < 0 or (abs(axis[1]) < 1e-8 and axis[0] < 0):
            axis = -axis
        normal = np.array([-axis[1], axis[0]])
        t, s = (points-center) @ axis, (points-center) @ normal
        length, breadth = np.ptp(t), np.ptp(s)
        if length < 16 or length/max(breadth, 1) < 3:
            continue
        low, high = t.min()+.2*length, t.max()-.2*length
        cuts = np.linspace(low, high, 13)
        samples = []
        for left, right in zip(cuts[:-1], cuts[1:]):
            hit = (t >= left) & (t <= right)
            if hit.sum() >= 4:
                a, b = np.percentile(s[hit], [10, 90])
                samples.append([(left+right)/2, a-.5, b+.5])
        if len(samples) < 6:
            continue
        sample = np.array(samples)
        left, error_l = _side_fit(sample[:, 0], sample[:, 1])
        right, error_r = _side_fit(sample[:, 0], sample[:, 2])
        middle = (low+high)/2
        width = float((right-left) @ [middle, 1])
        end_widths = (right[0]-left[0])*np.array([low,high])+(right[1]-left[1])
        # A wedge of background at an ROI edge is not two sides of one shaft.
        if width <= 0 or end_widths.min() <= 0 or end_widths.max()/end_widths.min() > 2:
            continue
        lines = []
        for side in [left, right]:
            coords = np.array([low, high])
            xy = center+coords[:, None]*axis+(side[0]*coords+side[1])[:, None]*normal
            lines.append((xy+offset).tolist())
        # A same-color branch (e.g. pool water) can reach the ROI boundary while
        # both fitted shaft sides remain fully visible. Check the measured sides.
        local_lines = np.asarray(lines)-offset
        clipped = bool((local_lines <= .5).any()
                       or (local_lines >= np.array(mask.shape[::-1])-1.5).any())
        reference = center+middle*axis+float((left+right) @ [middle, 1]/2)*normal+offset
        result = dict(edge_lines_px=lines, measured_center_px=reference.tolist(),
                      width_px=width, visible_length_px=float(length), width_clipped=clipped,
                      width_uncertainty_px=max(2., error_l+error_r))
        score = length*min(width, 30)
        if best is None or score > best[0]:
            best = (score, result)
    return None if best is None else best[1]


def _cylinder_position(edge_lines_px, reference_px, K, diameter_m):
    """Recover a visible axis section from the cylinder's two tangent planes.

    Uses calibrated angular width, including off-axis perspective and unequal
    fx/fy. It assumes a straight circular cylinder and two actual silhouette sides.
    """
    normals = []
    ray = np.linalg.solve(K, [*reference_px, 1.])
    for side in edge_lines_px:
        line = np.cross([*side[0], 1.], [*side[1], 1.])
        normal = K.T @ line
        length = np.linalg.norm(normal)
        if length < 1e-12:
            raise ValueError("Degenerate shaft edge")
        normal /= length
        normals.append(normal if normal @ ray > 0 else -normal)
    bisector = normals[0]+normals[1]
    if bisector @ bisector < 1e-12:
        raise ValueError("Unresolved angular shaft width")
    closest = diameter_m*bisector/(bisector @ bisector)
    axis = np.cross(*normals)
    axis /= max(np.linalg.norm(axis), 1e-15)
    axis_plane = np.cross(closest, axis)
    line = np.linalg.solve(K.T, axis_plane)
    if np.linalg.norm(line[:2]) < 1e-12:
        raise ValueError("Shaft viewed along its axis")
    line /= np.linalg.norm(line[:2])
    center = np.array(reference_px)-(line @ [*reference_px, 1.])*line[:2]
    ray = np.linalg.solve(K, [*center, 1.])
    system = np.column_stack((ray, -axis))
    if np.linalg.cond(system) > 1e6:
        raise ValueError("Shaft axis cannot be localized from this view")
    depth, _ = np.linalg.lstsq(system, closest, rcond=None)[0]
    xyz = depth*ray
    if not np.isfinite(xyz).all() or xyz[2] <= 0:
        raise ValueError("Cylinder is behind the camera")
    return xyz, center, axis


def _shaft_result(measurement, K, diameter_m):
    result = {**measurement, "center_px": measurement["measured_center_px"],
              "xyz_m": None, "distance_m": None, "status": "pixel_only"}
    if measurement["width_clipped"]:
        result.update(status="shaft_width_clipped",center_px=None,
                      candidate_center_px=measurement["measured_center_px"])
    elif measurement["width_px"] < 3:
        result["status"] = "shaft_too_thin"
    elif diameter_m is not None:
        try:
            xyz, center, axis = _cylinder_position(measurement["edge_lines_px"], result["center_px"], K, diameter_m)
            result.update(xyz_m=xyz.tolist(), center_px=center.tolist(),
                          distance_m=float(np.linalg.norm(xyz)), axis_unit=axis.tolist(),
                          axis_sign_ambiguous=True, status="estimated",
                          geometry_method="cylinder_tangent_planes",
                          relative_width_uncertainty=measurement["width_uncertainty_px"]/measurement["width_px"])
        except ValueError as exc:
            result.update(status="shaft_geometry_failed", geometry_status=str(exc))
    return result
