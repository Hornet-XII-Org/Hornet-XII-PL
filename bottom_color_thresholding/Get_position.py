"""Offline object position estimation from BGR images and pixel bounding boxes."""

import math

import cv2
import numpy as np


def get_ball_position(image, bbox, *, min_area=20, min_circularity=0.75,
                      canny_low=50, canny_high=150, blur_kernel=5,
                      camera=None, diameter_m=None, diameter_assumed=True,
                      geometry=None):
    """Find a circular candidate; metric output uses the near-axis approximation."""
    if (not math.isfinite(min_area) or min_area <= 0 or not 0 <= min_circularity <= 1
            or not 0 <= canny_low < canny_high <= 255 or not isinstance(blur_kernel, int)
            or blur_kernel < 1 or blur_kernel % 2 == 0):
        raise ValueError("Check area, circularity, Canny thresholds and the positive odd blur kernel")
    roi, (x1, y1, x2, y2) = _crop_roi(image, bbox)

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
                "radius_px": None, "xyz_m": None}

    _, u, v, radius = max(candidates, key=lambda item: item[0])
    center = (u + x1, v + y1)
    result = {"status": "candidate_needs_review", "center_px": center,
              "radius_px": radius, "xyz_m": None, "distance_m": None}
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


def _refine_rim(ellipse, strength, light, threshold):
    offsets = np.arange(-8, 9)
    for _ in range(2):
        rim = _ellipse_points(ellipse, 180)
        rotation = _rotation(ellipse[2])
        normal = ((rim-ellipse[0]) @ rotation / (np.array(ellipse[1])/2)**2) @ rotation.T
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
        samples = rim[:, None, :] + offsets[None, :, None]*normal[:, None, :]
        x, y = np.rint(samples[..., 0]).astype(int), np.rint(samples[..., 1]).astype(int)
        valid = (x >= 0) & (x < strength.shape[1]) & (y >= 0) & (y < strength.shape[0])
        x, y = np.clip(x, 0, strength.shape[1]-1), np.clip(y, 0, strength.shape[0]-1)
        value = np.minimum(strength[y, x]/threshold, 2) + 2*light[y, x]/255 - .025*np.abs(offsets)
        value[~valid] = -100
        indices = value.argmax(axis=1)
        points = samples[np.arange(180), indices][valid.any(axis=1)]
        if len(points) < 60:
            break
        try:
            updated = _fit_ellipse(points)
            if np.linalg.norm(np.array(updated[0])-ellipse[0]) > .1*min(ellipse[1]):
                break
            ellipse = updated
        except (ValueError, np.linalg.LinAlgError):
            break
    return ellipse


def _drum_observation(image, bbox, settings, object_mask=None):
    """Fit visible silhouette arcs, then verify the entire rim against image edges."""
    x1, y1, x2, y2 = bbox
    step = max(1, math.ceil(max(x2-x1, y2-y1) / settings.get("working_size", 640)))
    roi = image[y1:y2:step, x1:x2:step]
    rgb = roi[:, :, ::-1].astype(np.float32)
    if object_mask is not None:
        masks = [("provided_mask", object_mask[y1:y2:step, x1:x2:step])]
    else:
        red, green, blue = rgb.transpose(2, 0, 1)
        masks = [("red", (red > 1.25*green) & (red > 1.2*blue) & (red-green > 25)),
                 ("blue", (blue > red+14) & (green > red+6) & (blue > .95*green))]
    contours = []
    for source, candidate in masks:
        candidate = cv2.morphologyEx(candidate.astype(np.uint8)*255, cv2.MORPH_CLOSE,
                                     np.ones((3, 3), np.uint8), iterations=2)
        found, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours.extend((cv2.contourArea(c), source, c) for c in found)
    contours = [item for item in contours if item[0] >= max(80, .1*roi.shape[0]*roi.shape[1])]
    if not contours:
        return None
    _, source, contour = max(contours, key=lambda item: item[0])
    mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)  # Fill the silhouette, including holes.
    y, x = np.nonzero((mask > 0) & (cv2.erode(mask, None) == 0))
    boundary = np.column_stack((x, y)).astype(float)
    keep = (x > 3) & (x < mask.shape[1]-4) & (y > 3) & (y < mask.shape[0]-4)
    boundary = boundary[keep]
    if len(boundary) < 20:
        return None
    smooth = cv2.blur(rgb, (3, 3), borderType=cv2.BORDER_REPLICATE)
    gy, gx = np.gradient(smooth, axis=(0, 1))
    strength = np.sqrt((gx*gx+gy*gy).sum(axis=2))
    expanded = cv2.dilate(mask, None, iterations=4) > 0
    mask = mask > 0
    threshold = max(5., float(np.percentile(strength[mask], 80)))
    edges = (strength > threshold) & expanded
    distances = cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    light = np.minimum.reduce(rgb, axis=2)*mask
    # A few stray pixels near the crop edge do not justify extrapolating half a rim.
    clipped = max(mask[:4].sum(), mask[-4:].sum(), mask[:, :4].sum(), mask[:, -4:].sum()) >= .4*min(mask.shape)
    groups = [boundary]
    for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
        coordinate = (boundary-boundary.mean(axis=0)) @ [math.cos(angle), math.sin(angle)]
        groups.extend(boundary[coordinate <= np.quantile(coordinate, f)] for f in (.4, .55, .7, .85))
    best = None
    for points in groups:
        try:
            ellipse = _fit_ellipse(points)
        except (ValueError, np.linalg.LinAlgError):
            continue
        axes = ellipse[1]
        area = math.pi*np.prod(axes)/4
        if (min(axes) < 20 or min(axes)/max(axes) < .15
                or max(axes) > 2*math.hypot(*mask.shape) or not .2*mask.sum() < area < 2.5*mask.sum()):
            continue
        ellipse = _refine_rim(ellipse, strength, light, threshold)
        axes = ellipse[1]
        area = math.pi*np.prod(axes)/4
        rim = _ellipse_points(ellipse, 180)
        visible = ((rim >= [0, 0]).all(axis=1) & (rim < [mask.shape[1], mask.shape[0]]).all(axis=1))
        if visible.mean() < (.5 if clipped else .95):
            continue
        ix, iy = np.rint(rim[visible]).astype(int).T
        ix, iy = np.clip(ix, 0, mask.shape[1]-1), np.clip(iy, 0, mask.shape[0]-1)
        distance = np.minimum(distances[iy, ix], 12)
        support = float(np.mean(distance < 3))
        lip = light[iy, ix].mean()/255
        score = support-distance.mean()/20+.15*math.sqrt(min(area/mask.sum(), 1))+2*lip
        if support >= settings.get("min_edge_support", .60) and (best is None or score > best[0]):
            best = (score, ellipse, support)
    if best is None:
        return None
    return {"ellipse": best[1], "rim_support": best[2], "color": source if source in ("red", "blue") else None,
            "contour_source": source,
            "boundary": boundary, "strength": strength, "step": step, "offset": np.array([x1, y1])}


def _inner_rim(mouth, strength):
    """Track a visible inner ring, returning its fitted ellipse and image points."""
    center, axes, angle = mouth
    t, radius = np.linspace(0, 2*np.pi, 180, endpoint=False), np.linspace(.38, .88, 90)
    rotation = _rotation(angle)
    rays = np.column_stack((np.cos(t), np.sin(t))) * (np.array(axes)/2)
    points = (rays[:, None, :]*radius[None, :, None]) @ rotation.T + center
    x, y = np.rint(points[..., 0]).astype(int), np.rint(points[..., 1]).astype(int)
    visible = (x >= 0) & (x < strength.shape[1]) & (y >= 0) & (y < strength.shape[0])
    if visible.sum() < 100:
        return None
    values = strength[np.clip(y, 0, strength.shape[0]-1), np.clip(x, 0, strength.shape[1]-1)]
    scores = np.minimum(values/max(1., float(np.percentile(values[visible], 80))), 1)
    scores[~visible] = 0
    best, history = scores[0], []
    penalty = .06*(np.arange(90)[:, None]-np.arange(90)[None, :])**2
    for row in scores[1:]:
        transition = best[:, None]-penalty
        previous = transition.argmax(axis=0)
        best = row + transition[previous, np.arange(90)]
        history.append(previous)
    index, path = int(best.argmax()), []
    for previous in history[::-1]:
        path.append(index)
        index = int(previous[index])
    path = np.array([index] + path[::-1])
    keep = visible[np.arange(180), path] & (values[np.arange(180), path] > .5)
    observed = points[np.arange(180), path][keep]
    if len(observed) < 60:
        return None
    try:
        ellipse = _fit_ellipse(observed)
        local = _local_ellipse(observed, ellipse)
        residual = np.abs(np.linalg.norm(local, axis=1)-1)
        angles = np.sort(np.mod(np.arctan2(local[:, 1], local[:, 0]), 2*np.pi))
        largest_gap = np.diff(np.r_[angles, angles[0]+2*np.pi]).max()
        ratio = max(ellipse[1])/max(mouth[1])
        if residual.mean() < .06 and largest_gap < np.pi/3 and .4 < ratio < .9:
            return ellipse, observed
    except (ValueError, np.linalg.LinAlgError):
        pass
    return None


def _pose_direction(pose, K):
    """Derivative of the projected axis C + t*n at the mouth center."""
    x, y, z = pose["xyz_m"]
    nx, ny, nz = pose["normal_into_drum"]
    return np.array([K[0, 0]*(nx*z-x*nz), K[1, 1]*(ny*z-y*nz)])


def _select_drum_pose(poses, observation, K):
    """Try the exterior silhouette first; inspect the interior only if needed."""
    mouth, boundary = observation["ellipse"], observation["boundary"]
    outside = boundary[np.linalg.norm(_local_ellipse(boundary, mouth), axis=1) > 1.10]
    evidence = {"inner_search_used": False}

    def choose(cue, method):
        length = np.linalg.norm(cue)
        scores = []
        for pose in poses:
            direction = _pose_direction(pose, K)
            scores.append(float(direction @ cue / max(np.linalg.norm(direction)*length, 1e-9)))
        evidence["candidate_alignment"] = dict(zip((p["id"] for p in poses), scores))
        if length > 3 and max(scores) > .5 and (len(poses) == 1 or max(scores)-min(scores) > .4):
            evidence.update(cue_vector=cue, selection_method=method, selection_reliability="medium")
            return int(np.argmax(scores))
        return None

    # A visible exterior wall already resolves the inward direction. No inner fit.
    if len(outside) >= max(15, .08*len(boundary)):
        selected = choose(outside.mean(axis=0)-mouth[0], "observed_outer_wall")
        if selected is not None:
            evidence["wall_points"] = outside
            return selected, evidence
    if len(poses) == 1:
        evidence.update(selection_method="coincident_solutions", selection_reliability="high")
        return 0, evidence

    evidence["inner_search_used"] = True
    inner = _inner_rim(mouth, observation["strength"])
    if inner is not None:
        ellipse, points = inner
        evidence.update(bottom_ellipse=ellipse, bottom_points=points)
        selected = choose(np.array(ellipse[0])-mouth[0], "observed_inner_rim")
        if selected is not None:
            return selected, evidence
    # Automatic, explicitly low-confidence choice if image evidence is insufficient.
    selected = int(np.argmax([p["normal_into_drum"][2] for p in poses]))
    evidence.update(selection_method="optical_axis_prior_fallback", selection_reliability="low")
    return selected, evidence


def get_drum_position(image, bbox, *, camera=None, diameter_m=None,
                      diameter_assumed=True, detection=None, object_mask=None):
    """Return one selected circle pose; unknown-camera XYZ uses assumed 60 deg HFOV.

    XYZ is the mouth center in camera optical coordinates: right, down, forward.
    A known mouth diameter sets scale. No bottom radius or drum depth is used.
    object_mask: optional full-image binary silhouette from a segmentation model.
    Supplying it skips color thresholds; a bounding box alone is not a mask.
    """
    _, bbox = _crop_roi(image, bbox, min_size=20)
    settings = dict(detection or {})
    if not isinstance(settings.get("working_size", 640), int) or settings.get("working_size", 640) < 64:
        raise ValueError("working_size must be an integer of at least 64")
    if not 0 < settings.get("min_edge_support", .60) <= 1:
        raise ValueError("min_edge_support must be in (0, 1]")
    if diameter_m is None or not math.isfinite(diameter_m) or diameter_m <= 0:
        raise ValueError("Set the positive mouth diameter_m to calculate XYZ in metres")
    camera = camera_for_image(camera, image.shape)
    K = _camera_matrix(camera, image.shape)
    result = {"status": "no_mouth_found", "ellipse": None, "inner_ellipse": None,
              "center_px": None, "xyz_m": None, "distance_m": None,
              "pose_candidates": [], "selected_pose_id": None, "selection_method": None,
              "frame": "camera_optical", "target": "drum_mouth_center",
              "camera_used": camera, "parameters_assumed": _parameters_assumed(camera, diameter_assumed)}
    if object_mask is not None:
        object_mask = np.asarray(object_mask)
        if object_mask.shape != image.shape[:2] or not np.isfinite(object_mask).all():
            raise ValueError("object_mask must be a finite binary mask with the full image height and width")
        if not np.isin(object_mask, [0, 1, 255]).all():
            raise ValueError("object_mask must contain only 0/1 or 0/255")
        object_mask = object_mask != 0
    observation = _drum_observation(image, bbox, settings, object_mask)
    if observation is None:
        return result
    step, offset = observation["step"], observation["offset"]
    def original(ellipse):
        return [(np.array(ellipse[0])*step+offset).tolist(),
                (np.array(ellipse[1])*step).tolist(), ellipse[2]]
    mouth = original(observation["ellipse"])
    result.update(ellipse=mouth, ellipse_center_px=mouth[0],
                  rim_support=observation["rim_support"], color=observation["color"],
                  contour_source=observation["contour_source"])
    poses = _circle_poses_from_ellipse(mouth, K, diameter_m/2)
    if not poses:
        result["status"] = "geometry_failed"
        return result
    selected, evidence = _select_drum_pose(poses, observation, K)
    for i, pose in enumerate(poses):
        pose.update(selected=i == selected, distance_m=float(np.linalg.norm(pose["xyz_m"])),
                    image_alignment=evidence.get("candidate_alignment", {}).get(pose["id"]))
    chosen = poses[selected]
    reliability = evidence["selection_reliability"]
    result.update(status="estimated_with_prior" if reliability == "low" else
                  "estimated_assumed_parameters" if result["parameters_assumed"] else "estimated",
                  xyz_m=chosen["xyz_m"], center_px=chosen["center_px"],
                  distance_m=chosen["distance_m"], normal_into_drum=chosen["normal_into_drum"],
                  ray_unit=(np.array(chosen["xyz_m"])/chosen["distance_m"]).tolist(),
                  selected_pose_id=chosen["id"], selection_method=evidence["selection_method"],
                  selection_reliability=reliability, inner_search_used=evidence["inner_search_used"],
                  pose_candidates=poses, geometry_method="circle_pose_from_ellipse",
                  center_method="selected_circle_pose")
    if "wall_points" in evidence:
        result["wall_points_px"] = (evidence["wall_points"]*step+offset).tolist()
    if "bottom_ellipse" in evidence:
        inner = original(evidence["bottom_ellipse"])
        result.update(inner_ellipse=inner, bottom_ellipse_center_px=inner[0],
                      bottom_points_px=(evidence["bottom_points"]*step+offset).tolist())
    if "cue_vector" in evidence:
        cue = evidence["cue_vector"]
        result["observed_direction_unit"] = (cue/np.linalg.norm(cue)).tolist()
    direction = _pose_direction(chosen, K)
    length = np.linalg.norm(direction)
    if length > 1e-9:
        unit = direction/length
        result.update(image_axis_unit=unit.tolist(), image_axis_start_px=chosen["center_px"],
                      image_axis_end_px=(chosen["center_px"]+unit*.2*min(mouth[1])).tolist(),
                      image_axis_angle_deg=math.degrees(math.atan2(unit[1], unit[0])),
                      image_axis_method="projected_selected_normal")
    return result


def get_flare_position(image, bbox):
    raise NotImplementedError("Flare position estimation is not implemented")


def get_gate_position(image, bbox):
    raise NotImplementedError("Gate position estimation is not implemented")
