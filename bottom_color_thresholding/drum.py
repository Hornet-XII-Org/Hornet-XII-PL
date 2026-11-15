"""Drum mouth detection, observed bottom evidence and circle-pose selection."""

import math

import cv2
import numpy as np

from geometry_helpers import (
    _crop_roi, _search_roi, _camera_matrix, camera_for_image, _parameters_assumed,
    _rotation, _ellipse_points, _circle_poses_from_ellipse,
    _fit_ellipse, _local_ellipse,
)


def _refine_rim(ellipse, strength, light, threshold):
    reach = max(2,min(8,int(.035*max(ellipse[1]))))
    offsets = np.arange(-reach,reach+1)
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


def _small_rim_search(seed, rgb, mask, boundary):
    """Resolve a small mouth from a body-shaped seed using oriented image edges.

    A bounded coarse-to-fine ellipse grid is used only for small objects. Edge
    gradients should point across the rim; a bright wall/body outline alone is
    insufficient. No bottom size or metric pose is used in this image fit.
    """
    smooth=cv2.blur(rgb,(3,3),borderType=cv2.BORDER_REPLICATE)
    gy,gx=np.gradient(smooth,axis=(0,1))
    strength=np.sqrt((gx*gx+gy*gy).sum(axis=2))
    threshold=max(4.,float(np.percentile(strength[mask],70)))
    expanded=cv2.dilate(mask.astype(np.uint8),None,iterations=max(4,int(.15*seed[1][0])))>0
    angles=np.linspace(0,2*np.pi,96,endpoint=False)
    cosine,sine=np.cos(angles),np.sin(angles)
    minimum_major=.98*seed[1][0]
    if not (mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any()):
        # The outer mouth of an open bowl spans its visible body. Avoid selecting
        # a smaller inner highlight when a complete silhouette supplies the width.
        extent=np.ptp((boundary-seed[0])@_rotation(seed[2])[:,0])
        minimum_major=max(minimum_major,.97*extent)

    def choose(models):
        winner=None
        for start in range(0,len(models),256):
            model=np.asarray(models[start:start+256],float)
            u,v,a,b,angle=model.T
            rotation=np.deg2rad(angle);c,s=np.cos(rotation)[:,None],np.sin(rotation)[:,None]
            px=a[:,None]/2*cosine;py=b[:,None]/2*sine
            x=u[:,None]+c*px-s*py;y=v[:,None]+s*px+c*py
            nx=cosine/(a[:,None]/2);ny=sine/(b[:,None]/2)
            normal_x=c*nx-s*ny;normal_y=s*nx+c*ny
            norm=np.sqrt(normal_x**2+normal_y**2);normal_x/=norm;normal_y/=norm
            valid=(x>=0)&(x<mask.shape[1])&(y>=0)&(y<mask.shape[0])
            ix=np.clip(np.rint(x).astype(int),0,mask.shape[1]-1)
            iy=np.clip(np.rint(y).astype(int),0,mask.shape[0]-1)
            inside=expanded[iy,ix].mean(axis=1)
            edge=np.zeros_like(x)
            for shift in [-2,-1,0,1,2]:
                sx=np.clip(np.rint(x+shift*normal_x).astype(int),0,mask.shape[1]-1)
                sy=np.clip(np.rint(y+shift*normal_y).astype(int),0,mask.shape[0]-1)
                dx,dy=gx[sy,sx],gy[sy,sx]
                energy=(dx*dx+dy*dy).sum(axis=2)
                alignment=((dx*normal_x[...,None]+dy*normal_y[...,None])**2).sum(axis=2)/(energy+1e-9)
                edge=np.maximum(edge,np.minimum(np.sqrt(energy)/threshold,1)*alignment-.035*abs(shift))
            bx=boundary[None,:,0]-u[:,None];by=boundary[None,:,1]-v[:,None]
            radius=((bx*c+by*s)/(a[:,None]/2))**2+((-bx*s+by*c)/(b[:,None]/2))**2
            wall=(radius>1.1**2).mean(axis=1)
            light=np.max(rgb[iy,ix],axis=2).mean(axis=1)/255
            score=edge.mean(axis=1)+.4*light+.25*np.minimum(wall,.7)+.15*np.clip(a/seed[1][0]-1,0,.25)
            score[(valid.mean(axis=1)<.95)|(inside<.8)|(b<8)|(b>a)|(a<minimum_major)]=-100
            index=int(score.argmax())
            if winner is None or score[index]>winner[0]:
                winner=(float(score[index]),model[index])
        return winner

    center,axes,angle=seed
    rotation=_rotation(angle)
    models=[[center[0],center[1],axes[0],axes[1],angle]]
    for along in np.linspace(-.2,.2,5):
        for across in np.linspace(-.6,.6,9):
            origin=np.array(center)+np.array([along*axes[0],across*axes[1]])@rotation.T
            for scale in [.9,1.,1.1,1.2,1.35,1.5]:
                major=axes[0]*scale
                for ratio in [.10,.15,.20,.25,.35,.5,.75,1.]:
                    for turn in [-7.,0.,7.]:
                        models.append([*origin,major,max(8.,major*ratio),angle+turn])
    score,best=choose(models)
    local=[]
    for du in [-2.,0.,2.]:
        for dv in [-2.,0.,2.]:
            for da in [-2.,0.,2.]:
                for db in [-1.5,0.,1.5]:
                    for turn in [-3.,0.,3.]:
                        local.append(best+[du,dv,da,db,turn])
    refined_score,refined=choose(local)
    if refined_score>score:
        best=refined
    return [best[:2].tolist(),best[2:4].tolist(),float(best[4]%180)]


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
        blue_mask = (blue > red+14) & (green > red+6) & (blue > .95*green)
        if blue_mask.any():
            # Reject weak blue casts and dark shadow pixels relative to the bowl.
            chroma = (np.minimum(blue, green)-red)/(np.maximum(blue, green)+1)
            blue_mask &= chroma > max(0., float(np.percentile(chroma[blue_mask], 80))-.35)
            if blue_mask.any():
                blue_mask &= blue > .35*np.percentile(blue[blue_mask], 90)
        masks = [("red", (red > 1.25*green) & (red > 1.2*blue) & (red-green > 25)),
                 ("blue", blue_mask)]
        # Underwater blue casts may connect the bowl to the pool wall. Retain a
        # second, saturated core proposal instead of treating that wall as a bowl.
        if blue_mask.any():
            masks.append(("blue",blue_mask & (chroma>.25)))
    contours = []
    for source, candidate in masks:
        candidate = cv2.morphologyEx(candidate.astype(np.uint8)*255, cv2.MORPH_CLOSE,
                                     np.ones((3, 3), np.uint8), iterations=2)
        found, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours.extend((cv2.contourArea(c), source, c) for c in found)
    contours = [item for item in contours if item[0] >= max(80, .02*roi.shape[0]*roi.shape[1])]
    if not contours:
        return None
    def priority(item):
        area, _, points = item
        x,y,w,h=cv2.boundingRect(points)
        borders=sum([x<=1,y<=1,x+w>=roi.shape[1]-1,y+h>=roi.shape[0]-1])
        joined_background=(borders>=3 or min(w,h)/max(w,h)<.65 or max(w,h)*step<180)
        return area/(1+4*borders) if object_mask is None and joined_background else area
    _, source, contour = max(contours, key=priority)
    if not settings.get('_focused',False):
        # Derive processing scale/context from the object, not arbitrary box margins.
        bx,by,bw,bh = cv2.boundingRect(contour)
        margin = max(8,int((.30 if max(bw,bh)*step<180 else .10)*max(bw,bh)*step))
        focused = (max(x1,x1+bx*step-margin),max(y1,y1+by*step-margin),
                   min(x2,x1+(bx+bw)*step+margin),min(y2,y1+(by+bh)*step+margin))
        focus_step = max(1,math.ceil(max(focused[2]-focused[0],focused[3]-focused[1])/settings.get('working_size',640)))
        focus_area = (focused[2]-focused[0])*(focused[3]-focused[1])
        if focused != tuple(bbox) and (step > focus_step or (x2-x1)*(y2-y1) > 1.8*focus_area):
            return _drum_observation(image,focused,{**settings,'_focused':True},object_mask)
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
    bounds=cv2.boundingRect(contour)[2:]
    small = object_mask is None and max(bounds)*step < 180 and min(bounds)/max(bounds)<.8
    light = (np.maximum.reduce(rgb,axis=2) if small else np.minimum.reduce(rgb,axis=2))*mask
    # A few stray pixels near the crop edge do not justify extrapolating half a rim.
    clipped = max(mask[:4].sum(), mask[-4:].sum(), mask[:, :4].sum(), mask[:, -4:].sum()) >= .4*min(mask.shape)
    groups = [] if small else [boundary]
    for angle in ([] if small else np.linspace(0, 2*np.pi, 12, endpoint=False)):
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
        if expanded[iy, ix].mean() < .95:
            continue
        distance = np.minimum(distances[iy, ix], 12)
        support = float(np.mean(distance < 3))
        lip = light[iy, ix].mean()/255
        score = support-distance.mean()/20+.15*math.sqrt(min(area/mask.sum(), 1))+2*lip
        if support >= settings.get("min_edge_support", .60) and (best is None or score > best[0]):
            best = (score, ellipse, support)
    if small:
        # A small body is a more stable seed than whichever partial rim/foot arc
        # happened to win the proposal score in a slightly different crop.
        try:
            seed=_fit_ellipse(boundary)
            if min(seed[1])>=8 and max(seed[1])<180/step:
                fitted=_small_rim_search(seed,rgb,mask,boundary)
                rim=_ellipse_points(fitted,180)
                ix,iy=np.rint(rim).astype(int).T
                visible=(ix>=0)&(ix<mask.shape[1])&(iy>=0)&(iy<mask.shape[0])
                support=float((distances[iy[visible],ix[visible]]<3).mean()) if visible.any() else 0.
                if support>=settings.get("min_edge_support",.60):
                    best=(0.,fitted,support)
        except (ValueError,np.linalg.LinAlgError):
            pass
    if best is None:
        return None
    return {"ellipse": best[1], "rim_support": best[2], "color": source if source in ("red", "blue") else None,
            "contour_source": source,
            "boundary": boundary, "strength": strength, "step": step, "offset": np.array([x1, y1])}


def _inner_rim(mouth, strength, search_center=None):
    """Track a bottom rim around a search seed; the fitted center stays free."""
    center, axes, angle = mouth
    if not np.any(strength > .5):
        return None
    origin = np.asarray(center if search_center is None else search_center, dtype=float)
    t, radius = np.linspace(0, 2*np.pi, 180, endpoint=False), np.linspace(.05, .95, 90)
    rotation = _rotation(angle)
    directions = np.column_stack((np.cos(t), np.sin(t)))
    shift = _local_ellipse(origin[None, :], mouth)[0]
    if shift @ shift >= 1:
        return None
    # Distance from the shifted seed to the ORIGINAL mouth along each ray.
    along = directions @ shift
    limit = -along + np.sqrt(along**2 + 1 - shift @ shift)
    rays = directions * (np.array(axes)/2) * limit[:, None]
    points = (rays[:, None, :]*radius[None, :, None]) @ rotation.T + origin
    x, y = np.rint(points[..., 0]).astype(int), np.rint(points[..., 1]).astype(int)
    visible = (x >= 0) & (x < strength.shape[1]) & (y >= 0) & (y < strength.shape[0])
    distance = np.linalg.norm(_local_ellipse(points.reshape(-1, 2), mouth), axis=1).reshape(180, 90)
    visible &= distance < .90  # Exclude the mouth lip, using its unchanged ellipse.
    if visible.sum() < 100:
        return None
    values = strength[np.clip(y, 0, strength.shape[0]-1), np.clip(x, 0, strength.shape[1]-1)]
    scale = max(1., float(np.percentile(values[visible], 80)))
    scores = np.minimum(values/scale, 1)
    scores += .2*distance  # Mildly prefer the outer inner-ring over central markings.
    scores[~visible] = 0
    best, history = scores[0], []
    penalty = .06*(np.arange(90)[:, None]-np.arange(90)[None, :])**2
    # Repeat angular data so the retained middle turn has context across 0/360 deg.
    for row in np.tile(scores, (3, 1))[1:]:
        transition = best[:, None]-penalty
        previous = transition.argmax(axis=0)
        best = row + transition[previous, np.arange(90)]
        history.append(previous)
    index, path = int(best.argmax()), []
    for previous in history[::-1]:
        path.append(index)
        index = int(previous[index])
    path = np.array([index] + path[::-1])[180:360]
    if abs(int(path[0])-int(path[-1])) > 4:
        return None
    keep = visible[np.arange(180), path] & (values[np.arange(180), path] > .5)
    observed = points[np.arange(180), path][keep]
    if len(observed) < 60:
        return None
    try:
        ellipse = _fit_ellipse(observed)
        # Reject measured DP outliers, then refit; never draw invented rim samples.
        for _ in range(2):
            residual = np.abs(np.linalg.norm(_local_ellipse(observed, ellipse), axis=1)-1)
            inliers = residual < .05
            if inliers.all():
                break
            observed = observed[inliers]
            if len(observed) < 60:
                return None
            ellipse = _fit_ellipse(observed)
        local = _local_ellipse(observed, ellipse)
        residual = np.abs(np.linalg.norm(local, axis=1)-1)
        angles = np.sort(np.mod(np.arctan2(local[:, 1], local[:, 0]), 2*np.pi))
        largest_gap = np.diff(np.r_[angles, angles[0]+2*np.pi]).max()
        ratio = max(ellipse[1])/max(mouth[1])
        inside = np.linalg.norm(_local_ellipse(_ellipse_points(ellipse), mouth), axis=1) < .97
        if residual.mean() < .06 and largest_gap < np.pi/3 and .4 < ratio < .9 and inside.mean() > .95:
            rim = _ellipse_points(ellipse, 180)
            ix, iy = np.rint(rim).astype(int).T
            valid = (ix >= 0) & (ix < strength.shape[1]) & (iy >= 0) & (iy < strength.shape[0])
            support = np.zeros(180)
            support[valid] = cv2.dilate(strength, None)[iy[valid], ix[valid]]
            yy, xx = np.indices(strength.shape)
            interior = np.linalg.norm(_local_ellipse(np.column_stack((xx.ravel(), yy.ravel())), mouth), axis=1).reshape(strength.shape) < .90
            common_scale = max(1., float(np.percentile(strength[interior], 90)))
            quality = float(np.minimum(support/common_scale, 1).mean() - residual.mean())
            return ellipse, observed, quality
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

    evidence.update(inner_search_used=True, inner_search_centers=[])
    trials = []
    for pose in poses:
        direction = _pose_direction(pose, K)
        length = np.linalg.norm(direction)
        if length < 1e-9:
            continue
        unit = direction/length
        radius = 1/np.linalg.norm(unit @ _rotation(mouth[2]) / (np.array(mouth[1])/2))
        seed = np.array(mouth[0]) + unit*radius/3
        evidence["inner_search_centers"].append(seed)
        inner = _inner_rim(mouth, observation["strength"], seed)
        if inner is None:
            continue
        ellipse, points, quality = inner
        selected = choose(np.array(ellipse[0])-mouth[0], "observed_inner_rim")
        if selected is not None:
            trials.append((quality, selected, {**evidence, "bottom_ellipse": ellipse,
                                               "bottom_points": points}))
    if trials:
        trials.sort(key=lambda trial: trial[0], reverse=True)
        quality, selected, evidence = trials[0]
        # Two seeds may find the same rim. Conflicting, near-equal fits stay uncertain.
        if len(trials) == 1 or trials[1][1] == selected or quality-trials[1][0] > .05:
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
              "search_bbox_px":list(bbox),
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
        _, padded = _search_roi(image,bbox,min_size=20)
        if padded != bbox:
            observation = _drum_observation(image,padded,settings,object_mask)
            if observation is not None:
                result['search_bbox_px'] = list(padded)
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
    if "inner_search_centers" in evidence:
        result["inner_search_centers_px"] = (np.array(evidence["inner_search_centers"]).reshape(-1, 2)*step+offset).tolist()
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
