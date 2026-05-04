#!/usr/bin/env python3
"""AI-Based Detection of Low-Contrast Rivets - Midterm Baseline.

This script detects low-contrast rivets on a same-color 3D printed base.
It is designed for prototype images where the rivets are mostly circular and
mounted in a row on the front face of the blue base.

Pipeline:
1. Load image.
2. Resize for consistent processing.
3. Automatically find the blue object using HSV color thresholding.
4. Crop the front-face / rivet-row region of interest.
5. Enhance low-contrast edges using CLAHE and background correction.
6. Detect circular rivets using Hough Circle Transform.
7. Merge duplicate detections using non-maximum suppression.
8. Keep detections that lie on the rivet row.
9. Draw rivet boundaries and center points.

Install:
    pip install opencv-python numpy

Example:
    python detect_rivets.py --image "C:/Users/Will/Downloads/Photos-3-001/20260503_230647.jpg" --debug

Folder mode:
    python detect_rivets.py --folder "C:/Users/Will/Downloads/Photos-3-001" --output outputs --debug
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np


@dataclass
class Detection:
    x: float
    y: float
    r: float
    score: float
    method: str = "hough"


@dataclass
class DetectionResult:
    image_path: Path
    output_path: Path
    detections: list[Detection]
    roi_xywh: tuple[int, int, int, int]
    scale: float


def resize_keep_aspect(image: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    if w <= max_width:
        return image.copy(), 1.0
    scale = max_width / float(w)
    new_size = (max_width, int(round(h * scale)))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale


def find_blue_object_roi(
    image: np.ndarray,
    front_band: tuple[float, float] = (0.18, 0.88),
) -> tuple[tuple[int, int, int, int], np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([85, 35, 20], dtype=np.uint8)
    upper_blue = np.array([145, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = image.shape[:2]
        return (0, 0, w, h), mask

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    pad_x = int(0.03 * w)
    pad_y = int(0.03 * h)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(image.shape[1] - x, w + 2 * pad_x)
    h = min(image.shape[0] - y, h + 2 * pad_y)

    start_frac, end_frac = front_band
    y0 = y + int(start_frac * h)
    y1 = y + int(end_frac * h)
    y0 = max(0, min(image.shape[0] - 1, y0))
    y1 = max(y0 + 1, min(image.shape[0], y1))

    return (x, y0, w, y1 - y0), mask


def preprocess_roi(roi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(lightness)

    blur_size = max(31, int(round(min(roi.shape[:2]) * 0.25)) | 1)
    background = cv2.GaussianBlur(enhanced, (blur_size, blur_size), 0)
    corrected = cv2.addWeighted(enhanced, 1.5, background, -0.5, 0)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)

    blurred = cv2.GaussianBlur(corrected, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    return enhanced, corrected, blurred, edges


def circle_score(edges: np.ndarray, gray: np.ndarray, x: float, y: float, r: float) -> float:
    h, w = edges.shape[:2]
    x_i, y_i, r_i = int(round(x)), int(round(y)), int(round(r))

    if x_i - r_i - 3 < 0 or y_i - r_i - 3 < 0:
        return 0.0
    if x_i + r_i + 3 >= w or y_i + r_i + 3 >= h:
        return 0.0

    ring = np.zeros_like(edges)
    cv2.circle(ring, (x_i, y_i), r_i, 255, 2)
    ring_pixels = np.count_nonzero(ring)
    if ring_pixels == 0:
        return 0.0

    edge_pixels = np.count_nonzero(cv2.bitwise_and(edges, edges, mask=ring))
    edge_support = edge_pixels / float(ring_pixels)

    inner = np.zeros_like(gray)
    outer = np.zeros_like(gray)
    cv2.circle(inner, (x_i, y_i), max(1, int(0.65 * r_i)), 255, -1)
    cv2.circle(outer, (x_i, y_i), int(1.15 * r_i), 255, -1)
    cv2.circle(outer, (x_i, y_i), int(0.85 * r_i), 0, -1)

    inner_vals = gray[inner > 0]
    outer_vals = gray[outer > 0]
    contrast = 0.0 if len(inner_vals) == 0 or len(outer_vals) == 0 else abs(float(np.mean(inner_vals)) - float(np.mean(outer_vals))) / 255.0
    return float(0.8 * edge_support + 0.2 * contrast)


def hough_circle_candidates(
    blurred: np.ndarray,
    edges: np.ndarray,
    min_radius: int,
    max_radius: int,
    hough_votes: int,
    multi_pass: bool = False,
) -> list[Detection]:
    candidates: list[Detection] = []

    if multi_pass:
        vote_values = sorted(set([hough_votes - 4, hough_votes, hough_votes + 4, hough_votes + 8]))
        vote_values = [v for v in vote_values if v >= 8]
    else:
        vote_values = [hough_votes]

    for votes in vote_values:
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(25, int(min_radius * 2)),
            param1=90,
            param2=votes,
            minRadius=int(min_radius),
            maxRadius=int(max_radius),
        )

        if circles is None:
            continue

        for cx, cy, radius in np.round(circles[0]).astype(int):
            score = circle_score(edges, blurred, cx, cy, radius)
            candidates.append(Detection(float(cx), float(cy), float(radius), score, f"hough_p2_{votes}"))

    return candidates


def contour_ellipse_candidates(edges: np.ndarray, gray: np.ndarray, min_radius: int, max_radius: int) -> list[Detection]:
    candidates: list[Detection] = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    processed_edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(processed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < (min_radius**2) * 0.15 or area > (max_radius**2) * 6.0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_radius or h < min_radius or w > max_radius * 3 or h > max_radius * 3:
            continue
        aspect = w / float(h)
        if not (0.45 <= aspect <= 2.2) or len(contour) < 5:
            continue

        (cx, cy), (axis_a, axis_b), _angle = cv2.fitEllipse(contour)
        major = max(axis_a, axis_b)
        minor = min(axis_a, axis_b)
        if major < min_radius or major > max_radius * 2.5 or minor < min_radius * 0.6 or major / max(minor, 1.0) > 2.5:
            continue

        radius = (major + minor) / 4.0
        score = circle_score(edges, gray, cx, cy, radius) * 0.8
        candidates.append(Detection(float(cx), float(cy), float(radius), float(score), "ellipse"))

    return candidates


def non_max_suppression(candidates: Iterable[Detection], score_threshold: float, merge_factor: float, merge_min_distance: float) -> list[Detection]:
    kept: list[Detection] = []
    for candidate in sorted(candidates, key=lambda d: d.score, reverse=True):
        if candidate.score < score_threshold:
            continue
        duplicate = False
        for existing in kept:
            distance = np.hypot(candidate.x - existing.x, candidate.y - existing.y)
            merge_distance = max(merge_min_distance, merge_factor * (candidate.r + existing.r))
            if distance < merge_distance:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def filter_detections_by_row(detections: list[Detection], row_tolerance: Optional[float] = None) -> list[Detection]:
    if len(detections) < 4:
        return detections

    ys = np.array([d.y for d in detections], dtype=float)
    radii = np.array([d.r for d in detections], dtype=float)
    median_y = float(np.median(ys))
    median_r = float(max(10.0, np.median(radii)))

    coarse_limit = max(60.0, 2.0 * median_r)
    coarse = [d for d in detections if abs(d.y - median_y) <= coarse_limit]
    if len(coarse) < 4:
        coarse = detections

    xs2 = np.array([d.x for d in coarse], dtype=float)
    ys2 = np.array([d.y for d in coarse], dtype=float)

    try:
        slope, intercept = np.polyfit(xs2, ys2, 1)
    except Exception:
        return detections

    tolerance = row_tolerance if row_tolerance is not None else max(35.0, 1.35 * median_r)
    filtered = [d for d in detections if abs(d.y - (slope * d.x + intercept)) <= tolerance]
    return filtered if len(filtered) >= 3 else detections


def parse_roi_string(roi_text: str) -> tuple[int, int, int, int]:
    cleaned = roi_text.replace(",", " ")
    values = [int(float(v)) for v in cleaned.split()]
    if len(values) != 4:
        raise argparse.ArgumentTypeError("ROI must have four values: x y w h")
    x, y, w, h = values
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("ROI width and height must be positive")
    return x, y, w, h


def choose_roi_interactively(image: np.ndarray) -> tuple[int, int, int, int]:
    roi = cv2.selectROI("Select rivet row ROI, then press ENTER. Press ESC to cancel.", image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        raise RuntimeError("ROI selection was cancelled or invalid.")
    return x, y, w, h


def save_debug_images(debug_dir: Path, stem: str, blue_mask: np.ndarray, roi: np.ndarray, enhanced: np.ndarray, corrected: np.ndarray, edges: np.ndarray) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / f"{stem}_01_blue_mask.png"), blue_mask)
    cv2.imwrite(str(debug_dir / f"{stem}_02_roi.png"), roi)
    cv2.imwrite(str(debug_dir / f"{stem}_03_enhanced.png"), enhanced)
    cv2.imwrite(str(debug_dir / f"{stem}_04_corrected.png"), corrected)
    cv2.imwrite(str(debug_dir / f"{stem}_05_edges.png"), edges)


def annotate_image(image: np.ndarray, detections: list[Detection], roi_xywh: tuple[int, int, int, int], show_coords: bool) -> np.ndarray:
    output = image.copy()
    x0, y0, w, h = roi_xywh
    cv2.rectangle(output, (x0, y0), (x0 + w, y0 + h), (255, 255, 0), 2)

    for idx, d in enumerate(sorted(detections, key=lambda det: det.x), start=1):
        cx = int(round(x0 + d.x))
        cy = int(round(y0 + d.y))
        radius = int(round(d.r))
        cv2.circle(output, (cx, cy), radius, (0, 255, 0), 2)
        cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)

        label = f"{idx}:({cx},{cy})" if show_coords else str(idx)
        label_y = max(20, cy - radius - 6)
        cv2.putText(output, label, (max(0, cx - 10), label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

    return output


def write_centers_csv(csv_path: Path, results: list[DetectionResult]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "rivet_number", "center_x_resized", "center_y_resized", "center_x_original", "center_y_original", "radius_resized", "radius_original", "score", "method"])
        for result in results:
            x0, y0, _, _ = result.roi_xywh
            for idx, d in enumerate(sorted(result.detections, key=lambda det: det.x), start=1):
                cx_resized = x0 + d.x
                cy_resized = y0 + d.y
                writer.writerow([
                    result.image_path.name,
                    idx,
                    round(cx_resized, 2),
                    round(cy_resized, 2),
                    round(cx_resized / result.scale, 2),
                    round(cy_resized / result.scale, 2),
                    round(d.r, 2),
                    round(d.r / result.scale, 2),
                    round(d.score, 4),
                    d.method,
                ])


def detect_rivets_in_image(image_path: Path, output_path: Path, args: argparse.Namespace) -> DetectionResult:
    original = cv2.imread(str(image_path))
    if original is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image, scale = resize_keep_aspect(original, args.max_width)
    if args.roi is not None:
        roi_xywh = args.roi
        blue_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    elif args.select_roi:
        roi_xywh = choose_roi_interactively(image)
        blue_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    else:
        roi_xywh, blue_mask = find_blue_object_roi(image, front_band=(args.front_start, args.front_end))

    x0, y0, w, h = roi_xywh
    x0 = max(0, min(image.shape[1] - 1, x0))
    y0 = max(0, min(image.shape[0] - 1, y0))
    w = max(1, min(image.shape[1] - x0, w))
    h = max(1, min(image.shape[0] - y0, h))
    roi_xywh = (x0, y0, w, h)

    roi = image[y0 : y0 + h, x0 : x0 + w]
    enhanced, corrected, blurred, edges = preprocess_roi(roi)

    candidates = hough_circle_candidates(blurred, edges, args.min_radius, args.max_radius, args.hough_votes, args.multi_pass)
    if args.use_ellipse_fallback:
        candidates.extend(contour_ellipse_candidates(edges, blurred, args.min_radius, args.max_radius))

    detections = non_max_suppression(candidates, args.score_threshold, args.merge_factor, args.merge_min_distance)
    if not args.no_row_filter:
        detections = filter_detections_by_row(detections, row_tolerance=args.row_tolerance)
    detections = sorted(detections, key=lambda d: d.x)

    annotated = annotate_image(image, detections, roi_xywh, show_coords=args.show_coords)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)

    if args.debug:
        save_debug_images(output_path.parent / "debug", image_path.stem, blue_mask, roi, enhanced, corrected, edges)

    result = DetectionResult(image_path, output_path, detections, roi_xywh, scale)
    print_detection_summary(result)
    return result


def print_detection_summary(result: DetectionResult) -> None:
    x0, y0, _, _ = result.roi_xywh
    print(f"\nImage: {result.image_path.name}")
    print(f"Output: {result.output_path}")
    print(f"Detected rivets: {len(result.detections)}")
    if not result.detections:
        print("No rivets detected. Try lowering --hough-votes or using --select-roi.")
        return

    print("# | center resized (x,y) | center original (x,y) | radius original | score")
    print("--|----------------------|-----------------------|-----------------|------")
    for idx, d in enumerate(sorted(result.detections, key=lambda det: det.x), start=1):
        cx_resized = x0 + d.x
        cy_resized = y0 + d.y
        print(f"{idx:2d}| ({cx_resized:7.1f}, {cy_resized:7.1f}) | ({(cx_resized / result.scale):7.1f}, {(cy_resized / result.scale):7.1f}) | {(d.r / result.scale):7.1f} | {d.score:.3f}")


def image_files_from_folder(folder: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in extensions)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect and mark low-contrast rivet centers in still images.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Path to one input image.")
    source.add_argument("--folder", type=Path, help="Path to a folder of images.")

    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output file for --image mode, or output folder for --folder mode.")
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--min-radius", type=int, default=20)
    parser.add_argument("--max-radius", type=int, default=50)
    parser.add_argument("--hough-votes", type=int, default=22)
    parser.add_argument("--multi-pass", action="store_true")
    parser.add_argument("--score-threshold", type=float, default=0.06)
    parser.add_argument("--merge-factor", type=float, default=0.95)
    parser.add_argument("--merge-min-distance", type=float, default=35.0)
    parser.add_argument("--row-tolerance", type=float, default=None)
    parser.add_argument("--no-row-filter", action="store_true")
    parser.add_argument("--front-start", type=float, default=0.18)
    parser.add_argument("--front-end", type=float, default=0.88)
    parser.add_argument("--roi", type=parse_roi_string, default=None)
    parser.add_argument("--select-roi", action="store_true")
    parser.add_argument("--use-ellipse-fallback", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--show-coords", action="store_true")
    parser.add_argument("--csv", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.front_start < 0 or args.front_end > 1 or args.front_start >= args.front_end:
        raise ValueError("--front-start and --front-end must be fractions between 0 and 1, with start < end.")

    results: list[DetectionResult] = []
    if args.image is not None:
        input_path = args.image
        if not input_path.exists():
            raise FileNotFoundError(f"Input image does not exist: {input_path}")
        if args.output.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            output_path = args.output
        else:
            output_path = args.output / f"{input_path.stem}_detected.png"
        results.append(detect_rivets_in_image(input_path, output_path, args))
    else:
        input_folder = args.folder
        if not input_folder.exists():
            raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
        files = image_files_from_folder(input_folder)
        if not files:
            raise FileNotFoundError(f"No image files found in folder: {input_folder}")
        for image_path in files:
            results.append(detect_rivets_in_image(image_path, args.output / f"{image_path.stem}_detected.png", args))

    csv_path = args.csv if args.csv is not None else (results[0].output_path.with_name(f"{results[0].image_path.stem}_centers.csv") if args.image is not None else args.output / "rivet_centers.csv")
    write_centers_csv(csv_path, results)
    print(f"\nSaved center coordinates CSV: {csv_path}")


if __name__ == "__main__":
    main()
