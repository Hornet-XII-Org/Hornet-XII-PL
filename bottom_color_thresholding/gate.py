"""Gate corner detection, planar rectangle pose and partial-post measurements."""

import math

import cv2
import numpy as np

from geometry_helpers import (
    _crop_roi, _search_roi, _camera_matrix, camera_for_image, _parameters_assumed,
    _mask, _color_mask, _shaft, _shaft_result, _side_fit,
)


def _validate_quad(corners):
    points = np.asarray(corners, dtype=float)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("corners_px must be four finite points in TL, TR, BR, BL order")
    sides = np.roll(points, -1, axis=0)-points
    cross = sides[:, 0]*np.roll(sides[:, 1], -1)-sides[:, 1]*np.roll(sides[:, 0], -1)
    if np.min(np.linalg.norm(sides, axis=1)) < 4 or not (np.all(cross > 1e-5) or np.all(cross < -1e-5)):
        raise ValueError("Gate corners must form a nondegenerate convex quadrilateral")
    return points


def _diagonal_center(corners):
    corners = _validate_quad(corners)
    first = np.cross([*corners[0], 1.], [*corners[2], 1.])
    second = np.cross([*corners[1], 1.], [*corners[3], 1.])
    intersection = np.cross(first, second)
    if abs(intersection[2]) < 1e-9:
        raise ValueError("Gate diagonals have no finite intersection")
    return intersection[:2]/intersection[2]


def _gate_pose(corners, K, width_m, height_m, max_error=3.):
    corners = _validate_quad(corners)
    center = _diagonal_center(corners)
    model = np.array([[-width_m/2, -height_m/2, 0], [width_m/2, -height_m/2, 0],
                      [width_m/2, height_m/2, 0], [-width_m/2, height_m/2, 0]], dtype=float)
    ok, rvec, tvec = cv2.solvePnP(model, corners, K, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok or not np.isfinite(np.r_[rvec.ravel(), tvec.ravel()]).all():
        raise ValueError("Planar gate pose could not be solved")
    rotation = cv2.Rodrigues(rvec)[0]
    camera_points = model @ rotation.T+tvec.ravel()
    if np.min(camera_points[:, 2]) <= 0:
        raise ValueError("Gate pose is behind the camera")
    projected = camera_points @ K.T
    projected = projected[:, :2]/projected[:, 2, None]
    error = float(np.sqrt(np.mean(np.sum((projected-corners)**2, axis=1))))
    if error > max_error:
        raise ValueError("Corners do not fit the configured gate rectangle")
    normal = rotation[:, 2]
    if normal @ tvec.ravel() < 0:
        normal = -normal
    # Intersect the diagonal-center ray with the fitted gate plane.
    ray = np.linalg.solve(K, [*center, 1.])
    denominator = normal @ ray
    if abs(denominator) < 1e-8:
        raise ValueError("Gate plane is viewed edge-on")
    xyz = ray*float(normal @ tvec.ravel())/denominator
    return dict(center_px=center.tolist(), diagonal_center_px=center.tolist(), xyz_m=xyz.tolist(),
                distance_m=float(np.linalg.norm(xyz)), plane_normal_away=normal.tolist(),
                corners_px=corners.tolist(), reprojection_error_px=error,
                center_method="diagonal_intersection", geometry_method="planar_rectangle_ippe",
                normal_yaw_deg=math.degrees(math.atan2(normal[0], normal[2])),
                normal_elevation_deg=math.degrees(math.atan2(-normal[1], math.hypot(normal[0], normal[2]))),
                rotation_vector_rad=rvec.ravel().tolist(), orientation_status="single_planar_pose_estimate")


def _gate_quads(roi, mask, kind):
    """Propose perspective quads; never replace the measured shape by a bbox."""
    smooth = cv2.GaussianBlur(roi, (5, 5), 0)
    # A red/green stripe can have little grayscale contrast but a strong color edge.
    edges = np.maximum.reduce([cv2.Canny(smooth[:, :, c], 40, 120) for c in range(3)])
    source = edges if mask is None else mask
    contours, _ = cv2.findContours(source, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    distance = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    colors = ([_color_mask(roi, "red"), _color_mask(roi, "green")] if kind == "main"
              else [_color_mask(roi, "orange")]*2) if mask is None else None
    if colors is not None:
        colors = [cv2.dilate(c, None, iterations=4) for c in colors]
    proposals = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:40]:
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        for fraction in [.01, .025]:
            quad = cv2.approxPolyDP(hull, fraction*perimeter, True).reshape(-1, 2).astype(float)
            if len(quad) != 4 or cv2.contourArea(quad.astype(np.float32)) < .04*source.size:
                continue
            quad = quad[np.argsort(np.arctan2(*(quad-quad.mean(axis=0))[:, ::-1].T))]
            quad = np.roll(quad, -int(np.argmin(quad.sum(axis=1))), axis=0)
            if (quad <= 2).any() or (quad >= np.array(source.shape[::-1])-3).any():
                continue  # A clipped end is not a measured gate corner.
            try:
                _validate_quad(quad)
            except ValueError:
                continue
            lengths = np.linalg.norm(np.roll(quad, -1, axis=0)-quad, axis=1)
            if min(lengths[::2].mean(), lengths[1::2].mean())/max(lengths[::2].mean(), lengths[1::2].mean()) < .15:
                continue  # A nearly edge-on frame is not a reliable automatic full-gate detection.
            support = []
            for a, b in zip(quad, np.roll(quad, -1, axis=0)):
                points = a+(b-a)*np.linspace(.05, .95, 60)[:, None]
                x, y = np.rint(points).astype(int).T
                support.append(float(np.mean(distance[y, x] < 3)))
            if sum(value > .6 for value in support) < 3:
                continue
            if mask is not None:
                filled = np.zeros_like(mask)
                cv2.drawContours(filled, [quad.astype(np.int32).reshape(-1, 1, 2)], -1, 255, -1)
                if (mask[filled > 0] > 0).mean() > .55:
                    continue  # One solid pole does not form a gate opening.
            else:
                # Validate colored markings on opposite posts, allowing either view direction.
                found = []
                for a, b in [(quad[0], quad[3]), (quad[1], quad[2])]:
                    points = a+(b-a)*np.linspace(.1, .9, 80)[:, None]
                    x, y = np.rint(points).astype(int).T
                    votes = [float((c[y, x] > 0).mean()) for c in colors]
                    found.append(votes)
                if max(min(found[0][0], found[1][1]), min(found[0][1], found[1][0])) < .08:
                    continue
            score = sum(sorted(support, reverse=True)[:3])/3+.1*math.sqrt(cv2.contourArea(quad.astype(np.float32))/source.size)
            proposals.append((score, quad))
    return sorted(proposals, key=lambda item: item[0], reverse=True)


def _line(points):
    a, b = points
    line = np.cross([*a, 1.], [*b, 1.])
    return line/max(np.linalg.norm(line[:2]), 1e-12)


def _merge_lines(segments):
    groups = []
    for segment in sorted(segments, key=lambda p: -np.linalg.norm(p[1]-p[0])):
        line = _line(segment)
        for group in groups:
            if (abs(line[:2] @ group["line"][:2]) > .99
                    and np.max(abs(np.c_[segment, np.ones(2)] @ group["line"])) < 4):
                group["points"].extend(segment)
                break
        else:
            groups.append(dict(line=line, points=list(segment)))
    merged = []
    for group in groups:
        points = np.array(group["points"])
        center = points.mean(axis=0)
        _, _, vectors = np.linalg.svd(points-center, full_matrices=False)
        direction = vectors[0]
        extent = (points-center) @ direction
        ends = center+np.array([extent.min(), extent.max()])[:, None]*direction
        merged.append(dict(line=_line(ends), segment=ends, length=float(np.ptp(extent))))
    return sorted(merged, key=lambda item: -item["length"])


def _line_support(mask, a, b):
    points = np.rint(a+(b-a)*np.linspace(0, 1, 100)[:, None]).astype(int)
    valid = (points >= 0).all(axis=1) & (points < mask.shape[::-1]).all(axis=1)
    return float((mask[points[valid, 1], points[valid, 0]] > 0).mean()) if valid.sum() > 10 else 0.


def _refine_frame_side(segment, mask, vertical, outer):
    """Refit one actual outer edge; skip wide branches crossing a narrow pole."""
    primary, secondary = (1, 0) if vertical else (0, 1)
    a, b = segment
    if abs(b[primary]-a[primary]) < 1:
        return segment
    radius = max(8, int(.06*min(mask.shape)))
    samples = []
    for t in np.unique(np.linspace(min(a[primary], b[primary]), max(a[primary], b[primary]), 70).astype(int)):
        predicted = a[secondary]+(t-a[primary])*(b[secondary]-a[secondary])/(b[primary]-a[primary])
        low, high = int(round(predicted))-radius, int(round(predicted))+radius+1
        if t < 0 or t >= mask.shape[secondary] or low < 0 or high > mask.shape[primary]:
            continue
        row = mask[t, low:high] if vertical else mask[low:high, t]
        changes = np.diff(np.r_[False, row > 0, False].astype(int))
        runs = [(i,j) for i,j in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1))
                if i > 0 and j < len(row) and 2 <= j-i <= radius]
        if not runs:
            continue
        start, end = min(runs,key=lambda r:abs(low+(r[0]+r[1]-1)/2-predicted))
        samples.append([t,low+start-.5 if outer < 0 else low+end-.5])
    if len(samples) < 16:
        return segment
    samples = np.array(samples)
    fit, error = _side_fit(samples[:, 0], samples[:, 1])
    if error > 2.5:
        return segment
    result = np.zeros((2,2))
    result[:, primary] = samples[[0,-1], 0]
    result[:, secondary] = result[:, primary]*fit[0]+fit[1]
    return result


def _snap_frame_edge(segment, gx, gy, distance):
    """Move a color proposal onto a coherent physical image edge."""
    a,b=segment;length=np.linalg.norm(b-a)
    direction=(b-a)/max(length,1e-9);normal=np.array([-direction[1],direction[0]])
    t=np.linspace(.05*length,.95*length,80)
    shifts=np.arange(-8,9)
    points=a+t[:,None,None]*direction+shifts[None,:,None]*normal
    x,y=np.rint(points).astype(int).transpose(2,0,1)
    valid=(x>=0)&(x<distance.shape[1])&(y>=0)&(y<distance.shape[0])
    x=np.clip(x,0,distance.shape[1]-1);y=np.clip(y,0,distance.shape[0]-1)
    dx,dy=gx[y,x],gy[y,x]
    cross=((dx*normal[0]+dy*normal[1])**2).sum(axis=2)
    alignment=cross/((dx*dx+dy*dy).sum(axis=2)+1e-9)
    scores=np.minimum(np.sqrt(cross)/4.,2)*alignment-.08*abs(shifts)
    scores[~valid]=-100
    index=scores.argmax(axis=1)
    keep=scores[np.arange(len(t)),index]>.35
    if keep.sum()<20 or np.ptp(t[keep])<.3*length:
        return segment
    selected=shifts[index][keep]
    fit,error=_side_fit(t[keep],selected)
    if error>2.5:
        return segment
    candidate=np.array([a+fit[1]*normal,b+(fit[0]*length+fit[1])*normal])
    def support(line):
        p=line[0]+(line[1]-line[0])*np.linspace(.05,.95,80)[:,None]
        xx,yy=np.rint(p).astype(int).T
        good=(xx>=0)&(xx<distance.shape[1])&(yy>=0)&(yy<distance.shape[0])
        return float((distance[yy[good],xx[good]]<3).mean()) if good.sum()>30 else 0.
    return candidate if support(candidate)>support(segment)+.05 else segment


def _frame_lines(roi, mask, _dark=False):
    """Find a lintel joining two roughly upright posts, even with cropped feet.

    Color contrast guides this fallback; it does not require red/green posts.
    The three measured lines, rather than the ROI border, define the frame.
    """
    height, width = roi.shape[:2]
    if mask is None:
        hue, saturation, value = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).transpose(2, 0, 1)
        colored = (saturation > 45) & (value > 5)
        foreground = colored.copy()
        if colored.mean() > .25:
            dominant = np.argmax(np.bincount(hue[colored], minlength=180))
            difference = abs(hue.astype(float)-dominant)
            foreground &= np.minimum(difference, 180-difference) > 18
        foreground |= (saturation < 50) & (value > max(60., .75*np.percentile(value,90)))
        if _dark:
            local_light = cv2.GaussianBlur(value,(21,21),0).astype(float)
            foreground |= local_light-value > max(8.,.07*np.percentile(value,90))
        active_columns=np.percentile(value,99,axis=0)>8
        active_rows=np.percentile(value,99,axis=1)>8
        foreground &= active_columns[None,:]&active_rows[:,None]
    else:
        if (mask > 0).mean() > .55:
            return []  # A solid segmentation rectangle is not a gate opening.
        foreground = mask > 0
    edges = cv2.Canny(foreground.astype(np.uint8)*255, 40, 100)
    if mask is None:
        # A constant black video border also has a perfectly straight edge.
        # Remove its transition from Hough input, not just its filled pixels.
        active=active_columns[None,:]&active_rows[:,None]
        edges[cv2.erode(active.astype(np.uint8),None,iterations=3)==0]=0
    scale = min(width, height)
    raw = cv2.HoughLinesP(edges, 1, np.pi/720, max(15, int(.045*scale)),
                          minLineLength=max(20, .12*scale), maxLineGap=max(8, .05*scale))
    if raw is None:
        return _frame_lines(roi,mask,True) if mask is None and not _dark else []
    segments = raw.reshape(-1, 2, 2)
    posts = _merge_lines([s for s in segments if abs(s[1, 1]-s[0, 1]) > 1.7*abs(s[1, 0]-s[0, 0])])[:16]
    tops = _merge_lines([s for s in segments if abs(s[1, 0]-s[0, 0]) > 1.7*abs(s[1, 1]-s[0, 1])])[:20]
    nearby = cv2.dilate(foreground.astype(np.uint8)*255, None, iterations=2)
    photo_edges = (np.maximum.reduce([cv2.Canny(cv2.GaussianBlur(roi[:,:,c],(3,3),0),20,60)
                                    for c in range(3)]) if mask is None else edges)
    edge_distance = cv2.distanceTransform((photo_edges==0).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    gy,gx=np.gradient(cv2.GaussianBlur(roi,(3,3),0).astype(float),axis=(0,1))
    proposals = []
    for index, first in enumerate(posts):
        for second in posts[index+1:]:
            left, right = sorted([first, second], key=lambda p: p["segment"][:, 0].mean())
            if (right["segment"][:, 0].mean()-left["segment"][:, 0].mean() < .35*width
                    or min(left["length"], right["length"]) < .35*height
                    or abs(left["line"][:2] @ right["line"][:2]) < .94):
                continue
            for top in tops:
                intersections = [np.cross(p["line"], top["line"]) for p in [left, right]]
                if any(abs(p[2]) < 1e-8 for p in intersections):
                    continue
                corners = np.array([p[:2]/p[2] for p in intersections])
                a, b = corners
                if ((corners < [1, 1]).any() or (corners > [width-2, .65*height]).any()
                        or np.linalg.norm(b-a) < .35*width):
                    continue
                starts=np.array([p['segment'][:,1].min() for p in [left,right]])
                if (np.all(corners[:,1]-starts>.14*height)
                        or np.any(starts-corners[:,1]>.25*height)):
                    continue
                inside = _line_support(nearby, a, b)
                if inside < .65:
                    continue
                # Two sides and the cap of one thick pole do not enclose a gate opening.
                u,v = np.meshgrid(np.linspace(.15,.85,12),np.linspace(.15,.8,12))
                available = min(height-max(a[1],b[1])-2,left["length"],right["length"])
                opening = a+u[...,None]*(b-a)+np.stack((np.zeros_like(v),v*available),axis=-1)
                ox,oy = np.rint(opening).astype(int).reshape(-1,2).T
                if foreground[oy,ox].mean() > .4:
                    continue
                direction = (b-a)/np.linalg.norm(b-a)
                reach = max(12., .09*width)
                outside = (_line_support(nearby, a-reach*direction, a-.2*reach*direction)
                           + _line_support(nearby, b+.2*reach*direction, b+reach*direction))/2
                score = inside-.7*outside+.1*(min(left["length"], height)+min(right["length"], height))/height
                if score > .75:
                    # Refit the lintel BETWEEN its proposed joints. A collinear
                    # surface reflection outside the frame must not pull its fit.
                    proposals.append((score, corners, np.array([left["segment"], right["segment"], corners])))
    refined = []
    for score, corners, segments in sorted(proposals,key=lambda p:-p[0])[:32]:
        segments = np.array([_refine_frame_side(s,foreground,i<2,-1 if i!=1 else 1)
                             for i,s in enumerate(segments)])
        if mask is None:
            segments=np.array([_snap_frame_edge(s,gx,gy,edge_distance) for s in segments])
        joints = [np.cross(_line(s),_line(segments[2])) for s in segments[:2]]
        if any(abs(p[2]) < 1e-8 for p in joints):
            continue
        updated = np.array([p[:2]/p[2] for p in joints])
        if np.max(np.linalg.norm(updated-corners,axis=1)) < .05*min(width,height):
            along = updated[0]+(updated[1]-updated[0])*np.linspace(.05,.95,100)[:,None]
            x,y = np.rint(along).astype(int).T
            valid=(x>=0)&(x<width)&(y>=0)&(y<height)
            supported=(edge_distance[y[valid],x[valid]]<2.5)
            support = float(supported.mean()) if valid.sum()>50 else 0.
            ends=min(float(supported[:15].mean()),float(supported[-15:].mean())) if len(supported)>50 else 0.
            # A long real bar with an unsupported extension can have high average
            # support. Its two joints should have comparable edge contrast too.
            joint_contrast=0.
            if mask is None and valid.all():
                vector=updated[1]-updated[0]
                normal=np.array([-vector[1],vector[0]])/np.linalg.norm(vector)
                energy=np.zeros(len(along))
                for shift in [-3,-1,0,1,3]:
                    pixels=np.rint(along+shift*normal).astype(int)
                    px=np.clip(pixels[:,0],0,width-1);py=np.clip(pixels[:,1],0,height-1)
                    across=gx[py,px]*normal[0]+gy[py,px]*normal[1]
                    energy=np.maximum(energy,np.linalg.norm(across,axis=1))
                joint_contrast=min(np.median(energy[:15]),np.median(energy[-15:]))/max(5.,np.median(energy))
            if not _dark or support>.55:
                refined.append((score+.35*support+.30*ends+.30*min(joint_contrast,1.),updated,segments))
    if mask is None and not _dark:
        refined.extend(_frame_lines(roi,mask,True))
    refined.sort(key=lambda p:-p[0])
    return refined


def _pose_from_frame_lines(top_corners, post_segments, K, width_m, height_m):
    """Two parallel post directions + their perpendicular top edge set a pose.

    The known width fixes scale. The known height reconstructs the missing feet.
    A top edge on the post horizon is degenerate and supplies no unique depth.
    """
    vanishing = np.cross(*[_line(segment) for segment in post_segments])
    vertical = np.linalg.solve(K, vanishing)
    norm = np.linalg.norm(vertical)
    if norm < 1e-10:
        raise ValueError("Post directions are not independently measured")
    vertical /= norm
    rays = np.c_[top_corners, np.ones(2)] @ np.linalg.inv(K).T
    p, q = rays @ vertical
    if min(abs(p), abs(q)) < .025 or p*q <= 0:
        raise ValueError("Three-line gate geometry is degenerate near the post horizon")
    ratio = p/q
    depth = width_m/np.linalg.norm(ratio*rays[1]-rays[0])
    left, right = depth*rays[0], depth*ratio*rays[1]
    direction_pixel = K @ (left+.01*vertical)
    if (direction_pixel[:2]/direction_pixel[2]-top_corners[0])[1] < 0:
        vertical = -vertical
    world = np.array([left, right, right+height_m*vertical, left+height_m*vertical])
    if not np.isfinite(world).all() or world[:, 2].min() <= 0:
        raise ValueError("Reconstructed gate extends behind the camera")
    projected = world @ K.T
    pixels = projected[:, :2]/projected[:, 2, None]
    xyz = world.mean(axis=0)
    pixel = K @ xyz
    normal = np.cross((right-left)/width_m, vertical)
    if normal @ xyz < 0:
        normal = -normal
    return dict(center_px=(pixel[:2]/pixel[2]).tolist(), xyz_m=xyz.tolist(),
                distance_m=float(np.linalg.norm(xyz)), plane_normal_away=normal.tolist(),
                normal_yaw_deg=math.degrees(math.atan2(normal[0], normal[2])),
                normal_elevation_deg=math.degrees(math.atan2(-normal[1], math.hypot(normal[0], normal[2]))),
                observed_top_corners_px=top_corners.tolist(), model_corners_px=pixels.tolist(),
                bottom_corners_inferred=True, center_method="rectangle_from_three_frame_lines",
                geometry_method="parallel_posts_and_known_rectangle",
                orientation_status="single_pose_from_three_frame_lines")


def _gate_position_in_roi(image, bbox, *, camera=None, width_m=1.5, height_m=None,
                      dimensions_assumed=True, corners_px=None, object_mask=None,
                      post_diameter_m=None, kind="main", max_reprojection_error_px=3.):
    """Estimate a full rectangle, or report observed posts without inventing its center.

    Corners must use the SAME physical reference as width_m/height_m. The default
    is the outer frame extent; opening dimensions require matching inner corners.
    """
    roi, bbox = _crop_roi(image, bbox, min_size=10)
    if kind not in ("main", "qualification"):
        raise ValueError("gate kind must be main or qualification")
    height_m = (1.0 if kind == "qualification" else 1.5) if height_m is None else height_m
    if not all(math.isfinite(v) and v > 0 for v in [width_m, height_m, max_reprojection_error_px]):
        raise ValueError("Gate dimensions and reprojection tolerance must be positive")
    if post_diameter_m is not None and (not math.isfinite(post_diameter_m) or post_diameter_m <= 0):
        raise ValueError("Set a positive measured gate post_diameter_m, or null")
    camera = camera_for_image(camera, image.shape)
    K = _camera_matrix(camera, image.shape)
    mask = _mask(image, bbox, object_mask)
    result = dict(status="no_gate_found", center_px=None, xyz_m=None, distance_m=None,
                  target="gate_center", frame="camera_optical", camera_used=camera,
                  search_bbox_px=list(bbox),
                  dimensions_m=[float(width_m), float(height_m)],
                  parameters_assumed=_parameters_assumed(camera, dimensions_assumed))
    if corners_px is not None:
        corners = _validate_quad(corners_px)
        if (corners < bbox[:2]).any() or (corners >= bbox[2:]).any():
            raise ValueError("corners_px must lie within the original-image bbox")
        proposals = [(1., corners)]
    else:
        proposals = [(score, quad+np.array(bbox[:2])) for score, quad in _gate_quads(roi, mask, kind)]
    for score, corners in proposals[:12]:
        try:
            result.update(_gate_pose(corners, K, width_m, height_m, max_reprojection_error_px))
            result.update(status="estimated_assumed_parameters" if result["parameters_assumed"] else "estimated",
                          corner_source="provided" if corners_px is not None else "mask_hull" if mask is not None else "image_edges",
                          image_support=score)
            return result
        except (ValueError, cv2.error) as exc:
            if corners_px is not None:
                result.update(status="gate_pixel_only", center_px=_diagonal_center(corners).tolist(),
                              diagonal_center_px=_diagonal_center(corners).tolist(), corners_px=corners.tolist(),
                              center_method="diagonal_intersection", geometry_status=str(exc))
                return result
    # A cropped gate can still be resolved from three measured frame lines.
    frames = _frame_lines(roi, mask)
    if frames:
        score, top, segments = frames[0]
        offset = np.array(bbox[:2])
        top, segments = top+offset, segments+offset
        visible=segments.copy()
        for i in range(2):
            a,b=segments[i]
            low,high=max(top[i,1],min(a[1],b[1])),max(a[1],b[1])
            if high>low and abs(b[1]-a[1])>1:
                y=np.array([low,high])
                visible[i]=np.c_[a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1]),y]
        result.update(observed_frame_lines_px=visible.tolist(), image_support=score,
                      corner_source="frame_line_intersections")
        try:
            result.update(_pose_from_frame_lines(top, segments[:2], K, width_m, height_m))
            result["status"] = "estimated_partial_frame_assumed_parameters" if result["parameters_assumed"] else "estimated_partial_frame"
            return result
        except ValueError as exc:
            result.update(status="partial_gate", geometry_status=str(exc))
    posts = []
    colors = [None] if mask is not None else (["red", "green"] if kind == "main" else ["orange"])
    for color in colors:
        selected = mask if mask is not None else _color_mask(roi, color)
        measured = _shaft(selected, np.array(bbox[:2]), combine=True)
        if measured is not None:
            post = _shaft_result(measured, K, post_diameter_m)
            post.update(target="visible_gate_member_section", color=color, diameter_m=post_diameter_m,
                        parameters_assumed=True)
            posts.append(post)
    if posts:
        result.update(status="partial_gate", post_measurements=posts,
                      geometry_status="gate_center_unobservable_from_visible_posts")
    return result


def get_gate_position(image, bbox, *, camera=None, width_m=1.5, height_m=None,
                      dimensions_assumed=True, corners_px=None, object_mask=None,
                      post_diameter_m=None, kind="main", max_reprojection_error_px=3.):
    """Use bounded image context; keep explicit reference corners authoritative."""
    _, requested = _crop_roi(image,bbox,min_size=10)
    _, padded = _search_roi(image,requested,min_size=10)
    regions = [requested] if corners_px is not None or padded == requested else [padded,requested]
    best = None
    fitted=[]
    for region in regions:
        result = _gate_position_in_roi(image,region,camera=camera,width_m=width_m,height_m=height_m,
                    dimensions_assumed=dimensions_assumed,corners_px=corners_px,object_mask=object_mask,
                    post_diameter_m=post_diameter_m,kind=kind,max_reprojection_error_px=max_reprojection_error_px)
        result['input_bbox_px'] = list(requested)
        if result['xyz_m'] is not None:
            fitted.append(result)
            continue
        rank = (bool(result.get('observed_frame_lines_px')),len(result.get('post_measurements',[])))
        if best is None or rank > best[0]:
            best = (rank,result)
    if fitted:
        return max(fitted,key=lambda r:(r.get('corner_source') in ('provided','image_edges','mask_hull'),r.get('image_support',0)))
    return best[1]
