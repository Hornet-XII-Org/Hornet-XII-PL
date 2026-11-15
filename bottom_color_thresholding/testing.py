"""Run one image/ROI; show the chosen pose, measured rim and center in one window."""

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from Get_position import (get_ball_position, get_drum_position,
                          get_flare_position, get_gate_position)
from geometry_helpers import _ellipse_points, camera_for_image


def draw_result(image, bbox, result, diameter_m, camera, diameter_assumed=True):
    """Coordinates in result always refer to the original image, before the header."""
    picture = Image.fromarray(image[:, :, ::-1])
    draw = ImageDraw.Draw(picture)
    size = max(14, min(22, picture.width // 50))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    front = result.get('target', '').startswith(('gate_', 'visible_flare_'))
    draw.rectangle((bbox[0], bbox[1], bbox[2]-1, bbox[3]-1), outline=(245, 210, 30), width=2)
    if result.get('model_corners_px') is not None:
        corners = np.array(result['model_corners_px'])
        for a,b in zip(corners,np.roll(corners,-1,axis=0)):
            for t in np.arange(0.,1.,.05):
                draw.line([tuple(a+t*(b-a)),tuple(a+min(t+.025,1)*(b-a))],fill='orange',width=2)
    for segment in result.get('observed_frame_lines_px', []):
        draw.line([tuple(p) for p in segment],fill=(0,255,70),width=3)
    if result.get('corners_px') is not None:
        corners = [tuple(p) for p in result['corners_px']]
        draw.line(corners+[corners[0]], fill=(0,255,70), width=3)
        for i, point in enumerate(corners):
            draw.text(point, str(i+1), font=font, fill='white', stroke_width=1, stroke_fill='black')
        draw.line([corners[0], corners[2]], fill='white', width=1)
        draw.line([corners[1], corners[3]], fill='white', width=1)
    for item in [result]+result.get('post_measurements', []):
        if item.get('status') in ('shaft_width_clipped','shaft_too_thin','shaft_geometry_failed'):
            continue
        for side in item.get('edge_lines_px', []):
            draw.line([tuple(p) for p in side], fill=(0,255,70), width=3)
    for i, post in enumerate(result.get('post_measurements', [])):
        if post.get('center_px') is None:
            continue
        u,v = post['center_px']
        draw.line((u-7,v,u+7,v),fill='orange',width=3)
        draw.line((u,v-7,u,v+7),fill='orange',width=3)
        draw.text((u+8,v),f'P{i+1}',font=font,fill='orange',stroke_width=1,stroke_fill='black')
    for key, color in [("ellipse", (0, 255, 70))]:
        if result.get(key):
            points = _ellipse_points(result[key], 300)
            draw.line([tuple(p) for p in points] + [tuple(points[0])], fill=color, width=3)
    for point in result.get('wall_points_px', []) + result.get('bottom_points_px', []):
        u, v = point
        draw.ellipse((u-1, v-1, u+1, v+1), fill=(255, 160, 30))
    if result.get('image_axis_unit') is not None:
        start, end = np.array(result['image_axis_start_px']), np.array(result['image_axis_end_px'])
        direction = np.array(result['image_axis_unit'])
        side = np.array([-direction[1], direction[0]])
        draw.line([tuple(start), tuple(end)], fill='black', width=6)
        draw.line([tuple(start), tuple(end)], fill='white', width=2)
        draw.polygon([tuple(end), tuple(end-10*direction+4*side), tuple(end-10*direction-4*side)], fill='white')
    if result.get("radius_px") is not None and result.get("center_px") is not None:
        u, v = result["center_px"]
        r = result["radius_px"]
        draw.ellipse((u-r, v-r, u+r, v+r), outline=(0, 255, 70), width=3)
    for key, color, label in [("ellipse_center_px", (0, 220, 255), "E"),
                               ("bottom_ellipse_center_px", (255, 160, 30), "B"),
                               ("center_px", (255, 40, 40), "C")]:
        if result.get(key) is not None:
            u, v = result[key]
            draw.line((u-9, v, u+9, v), fill=(0, 0, 0), width=7)
            draw.line((u, v-9, u, v+9), fill=(0, 0, 0), width=7)
            draw.line((u-9, v, u+9, v), fill=color, width=3)
            draw.line((u, v-9, u, v+9), fill=color, width=3)
            label_xy = (u-24, v+10) if label == 'E' else (u+11, v-20)
            draw.text(label_xy, label, font=font, fill=color, stroke_width=1, stroke_fill=(0, 0, 0))
    chosen = result.get("selected_pose_id")
    kind = result.get('object_type','')
    lines = [(kind.upper()+' | ' if kind else '')+result['status'].upper()+(f' | SELECTED: {chosen}' if chosen else '')]
    if result.get('suggested_object'):
        lines.append('Check object type: image structure suggests '+result['suggested_object'])
    if result.get("selection_method"):
        method = {"observed_outer_wall": "outer wall (inner search skipped)",
                  "observed_inner_rim": "inner bottom contour",
                  "optical_axis_prior_fallback": "weak optical-axis prior",
                  "coincident_solutions": "coincident solutions"}[result["selection_method"]]
        lines.append(f'Evidence: {method} | confidence: {result["selection_reliability"]}')
    if result.get('dimensions_m'):
        width,height=result['dimensions_m']
        lines.append(f'Gate reference W x H = {width:.3f} x {height:.3f} m')
    else:
        diameter_m = result.get('diameter_m', diameter_m)
        lines.append(f'Input diameter D = {diameter_m:.3f} m ({"ASSUMED" if diameter_assumed else "measured"})')
    if result.get('width_px') is not None:
        lines.append(f'Shaft width = {result["width_px"]:.2f} px | target: visible axis section')
        lines.append(f'Flare color = {result.get("color")} | requested = {result.get("requested_color")}')
        if result.get('color_ambiguous'):
            lines.append('COLOR AMBIGUOUS: thin yellow/orange marker; check actual diameter/class')
        if result.get('measurement_source') == 'image_edge_pair':
            lines.append('Measured a local straight shaft section from two image edges')
        if result.get('measurement_reliability') == 'low':
            lines.append('LOW RESOLUTION: shaft only a few pixels wide')
    if result.get('reprojection_error_px') is not None:
        lines.append(f'Gate corner reprojection RMS = {result["reprojection_error_px"]:.2f} px')
    if result.get('normal_yaw_deg') is not None:
        lines.append(f'Gate normal: camera yaw = {result["normal_yaw_deg"]:.1f} deg | elevation = {result["normal_elevation_deg"]:.1f} deg')
    if result.get('bottom_corners_inferred'):
        lines.append('Gate reconstructed from top + two post directions and known W/H')
        lines.append('Orange dashed: model frame; bottom endpoints were not measured')
    if result.get('relative_width_uncertainty') is not None:
        lines.append(f'Approx. width sensitivity = {100*result["relative_width_uncertainty"]:.0f}% (pixel resolution only)')
    if result.get("ellipse"):
        a, b = result["ellipse"][1]
        u, v = result["ellipse_center_px"]
        lines.append(f'Rim axes = {a:.1f} x {b:.1f} px | ellipse center E = ({u:.1f}, {v:.1f}) px')
    if result.get("center_px") is not None:
        u, v = result["center_px"]
        label = 'Target center C' if front else 'Mouth/target center C'
        lines.append(f'{label} = ({u:.1f}, {v:.1f}) px (original image coordinates)')
    if result.get('bottom_ellipse_center_px') is not None:
        u, v = result['bottom_ellipse_center_px']
        lines.append(f'Bottom ellipse center B = ({u:.1f}, {v:.1f}) px')
    if result.get("xyz_m") is not None:
        x, y, z = result["xyz_m"]
        distance = result.get("distance_m", (x*x+y*y+z*z)**.5)
        lines.append(f'Camera XYZ = ({x:.3f}, {y:.3f}, {z:.3f}) m | range = {distance:.3f} m')
    else:
        lines.append('No XYZ: '+result.get('geometry_status', result['status']))
    if front:
        if result.get('status') == 'partial_gate':
            lines.append('Partial gate: gate center is not resolved from visible posts')
            for i,post in enumerate(result.get('post_measurements', [])):
                xyz = post.get('xyz_m')
                lines.append(f'Post P{i+1}: XYZ = {tuple(round(v,3) for v in xyz)} m' if xyz is not None
                             else f'Post P{i+1}: {post["status"]}; set measured post_diameter_m for scale')
        lines.append('Green: measured gate / shaft lines | red C: target | orange P: observed post')
        if result.get('color') == 'orange':
            lines.append('Orange flare is an obstacle: avoid contact')
    else:
        lines.append('Green: mouth | cyan E: ellipse center | red C: mouth center')
        lines.append('Orange: observed evidence'+(' / B: bottom center' if result.get('inner_ellipse') else '')+' | white: selected axis')
    camera = result.get('camera_used', camera)
    if camera and camera.get('mode') == 'estimate':
        lines.append(f'ASSUMED HFOV {camera["horizontal_fov_deg"]:g} deg | XYZ is an estimate, not calibrated')
    elif result.get("parameters_assumed"):
        lines.append('Camera / size / rectification unverified: XYZ is an estimate')
    if result.get('camera_status') == 'image_size_mismatch_estimate':
        lines.append('Configured camera size mismatch: used assumed FOV instead')
    if result.get("selection_reliability") == "low":
        lines.append('LOW EVIDENCE: automatic choice; orientation remains uncertain')
    wrapped = []
    for text in lines:
        line = ""
        for word in text.split():
            proposed = (line + " " + word).strip()
            if line and draw.textlength(proposed, font=font) > picture.width-20:
                wrapped.append(line)
                line = word
            else:
                line = proposed
        wrapped.append(line)
    height = (size+7)*len(wrapped)+16
    output = Image.new("RGB", (picture.width, picture.height+height), (20, 20, 20))
    output.paste(picture, (0, height))
    header = ImageDraw.Draw(output)
    for i, line in enumerate(wrapped):
        header.text((10, 7+i*(size+7)), line, font=font, fill=(245, 225, 130))
    return output, lines, height


def _pick_gate_corners(image):
    points=[]
    window='Click TL, TR, BR, BL | Enter use | R reset | Esc cancel'
    cv2.namedWindow(window,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window,min(1100,image.shape[1]),min(760,image.shape[0]))
    def click(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN and len(points)<4:
            points.append([x,y])
    cv2.setMouseCallback(window,click)
    while True:
        preview=Image.fromarray(image[:,:,::-1])
        draw=ImageDraw.Draw(preview)
        for i,(x,y) in enumerate(points):
            draw.ellipse((x-5,y-5,x+5,y+5),fill='red')
            draw.text((x+7,y),str(i+1),fill='white')
        cv2.imshow(window,np.array(preview)[:,:,::-1].copy())
        key=cv2.waitKey(50)&0xff
        if key in (10,13) and len(points)==4:
            cv2.destroyAllWindows()
            return points
        if key in (ord('r'),ord('R')):
            points.clear()
        if key==27 or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:
            cv2.destroyAllWindows()
            return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--image", type=Path)
    parser.add_argument("--mask", type=Path, help="Full-image binary object silhouette PNG")
    parser.add_argument("--object", choices=("ball", "drum", "flare", "gate"))
    parser.add_argument("--color", choices=("auto","red","yellow","blue","orange"), help="Flare color")
    parser.add_argument("--corners", type=float, nargs=8, help="Gate corners: TL TR BR BL in original-image pixels")
    parser.add_argument("--pick-corners", action="store_true", help="Click four gate reference corners")
    parser.add_argument("--bbox", type=int, nargs=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)
    try:
        config_path = args.config.expanduser().resolve()
        config = yaml.safe_load(config_path.read_text())
        test = config["testing"]
        target = args.object or test.get("object", "drum")
        if target not in ("ball", "drum", "flare", "gate"):
            raise ValueError("testing.object must be ball, drum, flare or gate")
        if target!='gate' and (args.corners is not None or args.pick_corners):
            raise ValueError('--corners/--pick-corners require --object gate')
        if args.color and target!='flare':
            raise ValueError('--color requires --object flare')
        filename = args.image or test.get("image")
        if not filename:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            try:
                filename = filedialog.askopenfilename(title=f"Choose a {target.upper()} test image")
            finally:
                root.destroy()
        if not filename:
            return None
        path = Path(filename).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() if args.image else config_path.parent) / path
        image = np.array(Image.open(path).convert("RGB"))[:, :, ::-1].copy()
        mask_path = args.mask or test.get("mask")
        object_mask = None
        if mask_path:
            if target == "ball":
                raise ValueError("--mask is supported for drum, flare and gate")
            mask_path = Path(mask_path).expanduser()
            if not mask_path.is_absolute():
                mask_path = (Path.cwd() if args.mask else config_path.parent) / mask_path
            object_mask = np.array(Image.open(mask_path).convert("L"))
        bbox = args.bbox if args.bbox is not None else test.get("bbox")
        if bbox is None:
            window = f"Select ONE {target.upper()}, including visible sides; Enter confirms"
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, min(1100, image.shape[1]), min(760, image.shape[0]))
            x, y, w, h = cv2.selectROI(window, image, showCrosshair=True)
            cv2.destroyAllWindows()
            if not w or not h:
                return None
            bbox = [x, y, x+w, y+h]
        obj = config["objects"][target]
        camera = config.get('camera')
        if camera is not None and not isinstance(camera, dict):
            raise ValueError('camera must be a mapping of intrinsics, or null for unknown-source photos')
        camera_status = 'configured' if camera is not None else 'assumed_fov'
        if camera is not None and camera.get('mode') != 'estimate' and any(camera.get(key) is not None and camera[key] != actual
                                     for key, actual in [('image_width', image.shape[1]), ('image_height', image.shape[0])]):
            camera, camera_status = None, 'image_size_mismatch_estimate'
        camera = camera_for_image(camera, image.shape)
        if camera.get('mode') == 'estimate' and camera_status == 'configured':
            camera_status = 'assumed_fov'
        diameter_m = obj.get('diameter_m')
        color = args.color or obj.get('color','auto')
        if target=='flare':
            diameter_m = None if color=='auto' else obj['orange_diameter_m'] if color=='orange' else obj['colored_diameter_m']
        common = dict(camera=camera, diameter_m=diameter_m,
                      diameter_assumed=obj.get("diameter_assumed", True))
        if target == "drum":
            result = get_drum_position(image, bbox, **common, detection=config.get("drum_detection", {}),
                                       object_mask=object_mask)
        elif target == "ball":
            result = get_ball_position(image, bbox, **common, **config["ball_detection"],
                                       geometry=config["ball_geometry"])
        elif target == 'flare':
            result = get_flare_position(image,bbox,**common,color=color,object_mask=object_mask,
                        colored_diameter_m=obj['colored_diameter_m'],orange_diameter_m=obj['orange_diameter_m'])
        else:
            corners = np.array(args.corners).reshape(4,2).tolist() if args.corners is not None else test.get('gate_corners_px')
            if args.pick_corners or test.get('pick_gate_corners',False):
                corners=_pick_gate_corners(image)
                if corners is None:
                    return None
            result = get_gate_position(image,bbox,camera=camera,width_m=obj['width_m'],height_m=obj['height_m'],
                        dimensions_assumed=obj.get('dimensions_assumed',True),corners_px=corners,
                        object_mask=object_mask,post_diameter_m=obj.get('post_diameter_m'),kind=obj.get('kind','main'),
                        max_reprojection_error_px=obj.get('max_reprojection_error_px',3.))
        result['camera_status'] = camera_status
        result['object_type'] = target
        overlay, summary, header_height = draw_result(image, bbox, result, diameter_m,
                                                     camera, obj.get("diameter_assumed", True))
        output = args.output.expanduser().resolve() if args.output else Path(tempfile.mkdtemp(prefix=f"hornet-{target}-"))
        if args.output:
            output.mkdir(parents=True, exist_ok=False)
        overlay.save(output / "overlay.jpg", quality=95)
        report = {"image": str(path), "mask": str(mask_path) if mask_path else None,
                  "bbox": list(bbox), "config": config,
                  "overlay_header_height_px": header_height, "result": result}
        (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except Exception as exc:
        cv2.destroyAllWindows()
        parser.error(str(exc))
    print("\n".join(summary))
    print(f"Saved to: {output}")
    if args.show:
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Result", min(1100, overlay.width), min(850, overlay.height))
        cv2.imshow("Result", np.array(overlay)[:, :, ::-1].copy())
        while cv2.waitKey(100) & 0xff not in (27, ord("q"), 13):
            try:
                if cv2.getWindowProperty("Result", cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
        cv2.destroyAllWindows()
    return report


if __name__ == "__main__":
    main()
