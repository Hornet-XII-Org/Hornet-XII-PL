"""Flare shaft detection and visible-axis-section position estimation."""

import math

import cv2
import numpy as np

from geometry_helpers import (
    _crop_roi, _search_roi, _camera_matrix, camera_for_image, _parameters_assumed,
    _mask, _color_mask, _shaft, _shaft_result,
)

_COLOR_HUES = {"red": 0, "yellow": 30, "blue": 110, "orange": 16}


def _hue_distance(hue, color):
    difference = abs(hue-_COLOR_HUES[color])
    return min(difference, 180-difference)


def _edge_shaft(roi, offset, color):
    """Measure a local straight section when color joins the shaft to its background."""
    height,width = roi.shape[:2]
    smooth = cv2.GaussianBlur(roi,(3,3),0)
    edges = np.maximum.reduce([cv2.Canny(smooth[:,:,c],20,60) for c in range(3)])
    scale = max(height,width)
    raw = cv2.HoughLinesP(edges,1,np.pi/720,max(15,int(.04*scale)),
                         minLineLength=max(25,.20*scale),maxLineGap=max(6,.035*scale))
    if raw is None:
        return None
    segments = []
    for segment in raw.reshape(-1,2,2):
        vector=segment[1]-segment[0];length=np.linalg.norm(vector)
        if length<.20*scale:
            continue
        if height>1.4*width and abs(vector[1])<.75*length:
            continue
        if width>1.4*height and abs(vector[0])<.75*length:
            continue
        segments.append((length,segment.astype(float)))
    segments=sorted(segments,key=lambda p:-p[0])[:40]
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    best=None
    for index,(length,first) in enumerate(segments):
        direction=(first[1]-first[0])/length
        normal=np.array([-direction[1],direction[0]])
        for other_length,second in segments[index+1:]:
            if abs(direction@((second[1]-second[0])/other_length))<.995:
                continue
            fits=[];ranges=[]
            for pair in [first,second]:
                t=pair@direction;s=pair@normal
                slope=(s[1]-s[0])/(t[1]-t[0])
                fits.append(np.array([slope,s[0]-slope*t[0]]));ranges.append([t.min(),t.max()])
            low=max(r[0] for r in ranges);high=min(r[1] for r in ranges)
            if high-low<.20*scale:
                continue
            middle=(low+high)/2
            left,right=sorted(fits,key=lambda f:f@[middle,1.])
            widths=(right[0]-left[0])*np.array([low,high])+right[1]-left[1]
            diameter=float((right-left)@[middle,1.])
            if widths.min()<2 or widths.max()/widths.min()>2 or diameter>min(.65*min(height,width),.25*(high-low)):
                continue
            t=np.linspace(low+.1*(high-low),high-.1*(high-low),50)
            side_left=t[:,None]*direction+(left[0]*t+left[1])[:,None]*normal
            side_right=t[:,None]*direction+(right[0]*t+right[1])[:,None]*normal
            center=(side_left+side_right)/2
            gap=max(2.,.4*diameter)
            points=np.stack((center,side_left-gap*normal,side_right+gap*normal),axis=1)
            x,y=np.rint(points).astype(int).transpose(2,0,1)
            valid=((x>=0)&(x<width)&(y>=0)&(y<height)).all(axis=1)
            if valid.sum()<30:
                continue
            pixels=roi[y[valid],x[valid]].astype(float)
            contrast=float(np.median(np.minimum(np.linalg.norm(pixels[:,0]-pixels[:,1],axis=1),
                                               np.linalg.norm(pixels[:,0]-pixels[:,2],axis=1))))
            if contrast<18:
                continue
            hue=float(np.median(hsv[y[valid,0],x[valid,0],0]))
            observed=min(_COLOR_HUES,key=lambda c:_hue_distance(hue,c))
            if color not in ('auto',None) and _hue_distance(hue,color)>30:
                continue
            cross_center=np.mean(center,axis=0)
            priority=1+abs(normal@(cross_center-np.array([width/2,height/2])))/max(1,min(width,height)/2)
            score=(high-low)*contrast/priority
            if best is None or score>best[0]:
                lines=np.array([side_left[[0,-1]],side_right[[0,-1]]])
                clipped=bool((lines<=.5).any() or (lines>=np.array([width,height])-1.5).any())
                measurement=dict(edge_lines_px=(lines+offset).tolist(),measured_center_px=(cross_center+offset).tolist(),
                    width_px=diameter,visible_length_px=float(high-low),width_clipped=clipped,
                    width_uncertainty_px=2.,measurement_source='image_edge_pair',edge_contrast=contrast)
                best=(score,observed if color=='auto' else color,measurement)
    return None if best is None else best[1:]


def get_flare_position(image, bbox, *, camera=None, diameter_m=None, color="auto",
                       diameter_assumed=True, object_mask=None,
                       colored_diameter_m=.016, orange_diameter_m=.15):
    """Return the visible shaft-axis section, NOT the golf ball or full-body center."""
    _, requested_bbox = _crop_roi(image,bbox,min_size=10)
    roi, bbox = _search_roi(image, requested_bbox, min_size=10)
    if color not in ("auto", "red", "yellow", "blue", "orange", None):
        raise ValueError("flare color must be auto, red, yellow, blue, orange or null with a mask")
    if color is None and (object_mask is None or diameter_m is None):
        raise ValueError("Set object_mask and diameter_m when flare color is unknown")
    sizes = [colored_diameter_m, orange_diameter_m]+([] if diameter_m is None else [diameter_m])
    if not all(math.isfinite(d) and d > 0 for d in sizes):
        raise ValueError("Flare diameters must be positive and finite")
    camera = camera_for_image(camera, image.shape)
    K = _camera_matrix(camera, image.shape)
    mask = _mask(image, bbox, object_mask)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    measurement, detected_color, source = None, color, "provided_mask"
    if mask is not None:
        measurement = _shaft(mask, np.array(bbox[:2]))
        if color == "auto" and measurement is not None:
            # In automatic mode the color supplies a nominal physical diameter.
            hue = hsv[:, :, 0][mask > 0]
            middle = float(np.median(hue))
            detected_color = min(_COLOR_HUES, key=lambda c: _hue_distance(middle, c))
    else:
        candidates = []
        for candidate_color in (["red", "yellow", "blue", "orange"] if color == "auto" else [color]):
            selected = _color_mask(roi, candidate_color)
            selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            observed = _shaft(selected, np.array(bbox[:2]))
            if observed is not None and observed['visible_length_px']>=.45*max(roi.shape[:2]):
                score = observed["visible_length_px"]*min(observed["width_px"], 30)
                # Overlapping hue ranges should prefer the actual central hue.
                hue = float(np.median(hsv[:, :, 0][selected > 0]))
                difference = _hue_distance(hue, candidate_color)
                candidates.append((not observed['width_clipped'],score/(1+difference/180),candidate_color,observed))
        if candidates:
            _, _, detected_color, measurement = max(candidates,key=lambda c:c[:2])
        if measurement is None or measurement['width_clipped']:
            edge_measurement=_edge_shaft(roi,np.array(bbox[:2]),color)
            if edge_measurement is not None:
                detected_color,measurement=edge_measurement
        if (color=='auto' and measurement is not None and detected_color=='orange'
                and measurement['visible_length_px']/measurement['width_px']>20):
            # A thin colored marker can look orange underwater. The nominal wide
            # orange obstacle has a different size class; this remains an assumption.
            detected_color='yellow'
            measurement['color_ambiguous']=True
        source = "color_"+str(detected_color)
    diameter_source = "explicit" if diameter_m is not None else "nominal_for_color"
    frame_conflict = False
    if measurement is not None and mask is None and roi.shape[1] > .45*roi.shape[0]:
        # A gate's top bar is also cylindrical. Geometry alone cannot call it a flare.
        # Only reject when the surrounding top + two posts form a coherent frame.
        from gate import _frame_lines
        frame_conflict = bool(_frame_lines(roi,None))
        if not frame_conflict and bbox != requested_bbox:
            frame_conflict = bool(_frame_lines(_crop_roi(image,requested_bbox)[0],None))
        if frame_conflict:
            measurement = None
    if diameter_m is None:
        diameter_m = orange_diameter_m if detected_color == "orange" else colored_diameter_m
    result = dict(status="no_shaft_found", center_px=None, xyz_m=None, distance_m=None,
                  target="visible_flare_shaft_section", frame="camera_optical", color=detected_color,
                  requested_color=color, contour_source=source,
                  search_bbox_px=list(bbox),
                  diameter_source=diameter_source,
                  diameter_m=float(diameter_m), camera_used=camera,
                  parameters_assumed=_parameters_assumed(camera, diameter_assumed))
    if frame_conflict:
        result.update(status='object_type_mismatch',suggested_object='gate',
                      geometry_status='Gate-like frame in ROI; select gate or isolate one flare shaft')
    elif measurement is not None:
        result.update(_shaft_result(measurement, K, diameter_m))
        if result["xyz_m"] is not None:
            result["status"] = "estimated_assumed_parameters" if result["parameters_assumed"] else "estimated"
            result["measurement_reliability"] = "low" if result["relative_width_uncertainty"] > .15 else "medium"
    elif mask is None:
        result["geometry_status"] = f"No shaft for color {color}; use color auto or the actual shaft color"
    return result
