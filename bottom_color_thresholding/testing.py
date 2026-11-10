"""Run one image/ROI; show the chosen pose, measured rim and center in one window."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from Get_position import cv2, get_ball_position, get_drum_position, _ellipse_points, camera_for_image


def draw_result(image, bbox, result, diameter_m, camera, diameter_assumed=True):
    """Coordinates in result always refer to the original image, before the header."""
    picture = Image.fromarray(image[:, :, ::-1])
    draw = ImageDraw.Draw(picture)
    size = max(14, min(22, picture.width // 50))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle((bbox[0], bbox[1], bbox[2]-1, bbox[3]-1), outline=(245, 210, 30), width=2)
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
    lines = [result['status'].upper()+(f' | SELECTED: {chosen}' if chosen else '')]
    if result.get("selection_method"):
        method = {"observed_outer_wall": "outer wall (inner search skipped)",
                  "observed_inner_rim": "inner bottom contour",
                  "optical_axis_prior_fallback": "weak optical-axis prior",
                  "coincident_solutions": "coincident solutions"}[result["selection_method"]]
        lines.append(f'Evidence: {method} | confidence: {result["selection_reliability"]}')
    lines.append(f'Input diameter D = {diameter_m:.3f} m ({"ASSUMED" if diameter_assumed else "measured"})')
    if result.get("ellipse"):
        a, b = result["ellipse"][1]
        u, v = result["ellipse_center_px"]
        lines.append(f'Rim axes = {a:.1f} x {b:.1f} px | ellipse center E = ({u:.1f}, {v:.1f}) px')
    if result.get("center_px") is not None:
        u, v = result["center_px"]
        lines.append(f'Mouth/target center C = ({u:.1f}, {v:.1f}) px (original image coordinates)')
    if result.get('bottom_ellipse_center_px') is not None:
        u, v = result['bottom_ellipse_center_px']
        lines.append(f'Bottom ellipse center B = ({u:.1f}, {v:.1f}) px')
    if result.get("xyz_m") is not None:
        x, y, z = result["xyz_m"]
        distance = result.get("distance_m", (x*x+y*y+z*z)**.5)
        lines.append(f'Camera XYZ = ({x:.3f}, {y:.3f}, {z:.3f}) m | range = {distance:.3f} m')
    else:
        lines.append('No XYZ: '+result.get('geometry_status', result['status']))
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--image", type=Path)
    parser.add_argument("--mask", type=Path, help="Drum silhouette: full-image binary 0/255 PNG")
    parser.add_argument("--object", choices=("ball", "drum"))
    parser.add_argument("--bbox", type=int, nargs=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)
    try:
        config_path = args.config.expanduser().resolve()
        config = yaml.safe_load(config_path.read_text())
        test = config["testing"]
        target = args.object or test.get("object", "drum")
        if target not in ("ball", "drum"):
            raise ValueError("testing.object must be ball or drum")
        filename = args.image or test.get("image")
        if not filename:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            try:
                filename = filedialog.askopenfilename(title="Choose a test image")
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
            if target != "drum":
                raise ValueError("--mask is supported for drum only")
            mask_path = Path(mask_path).expanduser()
            if not mask_path.is_absolute():
                mask_path = (Path.cwd() if args.mask else config_path.parent) / mask_path
            object_mask = np.array(Image.open(mask_path).convert("L"))
        bbox = args.bbox if args.bbox is not None else test.get("bbox")
        if bbox is None:
            window = "Select ONE object, including its wall; Enter confirms"
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
        common = dict(camera=camera, diameter_m=obj["diameter_m"],
                      diameter_assumed=obj.get("diameter_assumed", True))
        if target == "drum":
            result = get_drum_position(image, bbox, **common, detection=config.get("drum_detection", {}),
                                       object_mask=object_mask)
        else:
            result = get_ball_position(image, bbox, **common, **config["ball_detection"],
                                       geometry=config["ball_geometry"])
        result['camera_status'] = camera_status
        overlay, summary, header_height = draw_result(image, bbox, result, obj["diameter_m"],
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
