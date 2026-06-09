r"""
Multi-class rivet center + segmentation pipeline
------------------------------------------------
Designed for three Windows photo folders such as:
    C:\Users\Will\Downloads\Photos-3-circle
    C:\Users\Will\Downloads\Photos-3-oval
    C:\Users\Will\Downloads\Photos-3-damaged

Main commands:
    python rivet_pipeline_multiclass_autolabel.py label --out "C:\Users\Will\Downloads\rivet_project_shapes" ^
      --class_dir circle="C:\Users\Will\Downloads\Photos-3-circle" ^
      --class_dir oval="C:\Users\Will\Downloads\Photos-3-oval" ^
      --class_dir damaged="C:\Users\Will\Downloads\Photos-3-damaged"

    python rivet_pipeline_multiclass_autolabel.py make_bad_csv --out "C:\Users\Will\Downloads\rivet_project_shapes"
    python rivet_pipeline_multiclass_autolabel.py build --out "C:\Users\Will\Downloads\rivet_project_shapes" --target_train 1000 --overwrite
    python rivet_pipeline_multiclass_autolabel.py train --out "C:\Users\Will\Downloads\rivet_project_shapes" --epochs 80 --batch 4
    python rivet_pipeline_multiclass_autolabel.py eval  --out "C:\Users\Will\Downloads\rivet_project_shapes" --split test
    python rivet_pipeline_multiclass_autolabel.py live  --out "C:\Users\Will\Downloads\rivet_project_shapes" --source 0

The model learns:
    - class segmentation: background / circle / oval / damaged
    - class-specific center heatmaps: circle centers / oval centers / damaged centers

Dependencies:
    pip install opencv-python numpy torch tqdm
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except Exception:
    torch = None
    nn = None
    F = None
    Dataset = object
    DataLoader = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMG_W = 512
IMG_H = 384
DEFAULT_ORIGINAL_RADIUS = 34.0  # pixels in original photos; adjust during labeling with +/-.
RANDOM_SEED = 42

# Fixed class order for consistent model outputs.
SHAPE_CLASSES = ["circle", "oval", "damaged"]
CLASS_TO_ID = {name: i + 1 for i, name in enumerate(SHAPE_CLASSES)}  # background=0
ID_TO_CLASS = {i + 1: name for i, name in enumerate(SHAPE_CLASSES)}
NUM_FG_CLASSES = len(SHAPE_CLASSES)
NUM_SEG_CLASSES = NUM_FG_CLASSES + 1  # background + foreground classes

# BGR display colors for OpenCV overlays.
CLASS_COLORS = {
    "circle": (0, 255, 255),
    "oval": (0, 200, 0),
    "damaged": (0, 0, 255),
}


# -----------------------------
# File utilities
# -----------------------------

def ensure_dirs(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset").mkdir(exist_ok=True)
    for sub in ["images", "masks", "labels"]:
        for split in ["train", "val", "test"]:
            (out / "dataset" / sub / split).mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    (out / "reports").mkdir(exist_ok=True)


def list_images_recursive(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image folder: {image_dir}")
    files = [p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=lambda p: str(p.relative_to(image_dir)).lower())


def parse_class_dirs(items: Optional[List[str]]) -> Dict[str, Path]:
    """Parse repeated --class_dir class=C:/path arguments."""
    out: Dict[str, Path] = {}
    if not items:
        return out
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected --class_dir name=path, got: {item}")
        name, path = item.split("=", 1)
        name = name.strip().lower()
        if name not in CLASS_TO_ID:
            raise ValueError(f"Unknown class '{name}'. Use one of: {', '.join(SHAPE_CLASSES)}")
        out[name] = Path(path.strip().strip('"'))
    return out


def save_sources(out: Path, class_dirs: Dict[str, Path]) -> None:
    ensure_dirs(out)
    data = {name: str(path) for name, path in class_dirs.items()}
    with (out / "sources.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_sources(out: Path) -> Dict[str, Path]:
    src = out / "sources.json"
    if not src.exists():
        raise FileNotFoundError(
            f"Missing {src}. Run the label command with --class_dir circle=... --class_dir oval=... --class_dir damaged=..."
        )
    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)
    class_dirs: Dict[str, Path] = {}
    for name, path in data.items():
        name = name.lower()
        if name in CLASS_TO_ID:
            class_dirs[name] = Path(path)
    return class_dirs


def collect_image_records(class_dirs: Dict[str, Path]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for class_name in SHAPE_CLASSES:
        if class_name not in class_dirs:
            continue
        root = class_dirs[class_name]
        for path in list_images_recursive(root):
            rel = path.relative_to(root).as_posix()
            key = f"{class_name}/{rel}"
            records.append({
                "image_key": key,
                "class_name": class_name,
                "class_id": CLASS_TO_ID[class_name],
                "rel_path": rel,
                "path": path,
            })
    records.sort(key=lambda r: str(r["image_key"]).lower())
    return records


def load_labels(label_csv: Path) -> Dict[str, Dict[str, object]]:
    labels: Dict[str, Dict[str, object]] = {}
    if not label_csv.exists():
        return labels
    with label_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_key = row["image_key"]
            class_name = row["class_name"].lower()
            rel_path = row.get("rel_path", image_key.split("/", 1)[-1])
            x = float(row["x"])
            y = float(row["y"])
            r = float(row.get("radius", DEFAULT_ORIGINAL_RADIUS))
            if class_name not in CLASS_TO_ID:
                continue
            if image_key not in labels:
                labels[image_key] = {"class_name": class_name, "rel_path": rel_path, "points": []}
            labels[image_key]["points"].append((x, y, r))
    return labels


def save_labels(label_csv: Path, labels: Dict[str, Dict[str, object]]) -> None:
    label_csv.parent.mkdir(parents=True, exist_ok=True)
    with label_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["image_key", "class_name", "rel_path", "x", "y", "radius"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for image_key in sorted(labels.keys()):
            item = labels[image_key]
            class_name = str(item["class_name"])
            rel_path = str(item.get("rel_path", image_key.split("/", 1)[-1]))
            for x, y, r in item.get("points", []):
                writer.writerow({
                    "image_key": image_key,
                    "class_name": class_name,
                    "rel_path": rel_path,
                    "x": f"{float(x):.3f}",
                    "y": f"{float(y):.3f}",
                    "radius": f"{float(r):.3f}",
                })


def read_split_csv(path: Path) -> Dict[str, str]:
    split_map: Dict[str, str] = {}
    if not path.exists():
        return split_map
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split_map[row["image_key"]] = row["split"]
    return split_map


def write_split_csv(path: Path, split_map: Dict[str, str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_key", "split"])
        writer.writeheader()
        for key, split in sorted(split_map.items()):
            writer.writerow({"image_key": key, "split": split})


def truthy_text(value: object) -> bool:
    """Return True for CSV values that mean this image should be manually fixed."""
    txt = str(value).strip().lower()
    return txt in {"1", "true", "yes", "y", "bad", "fix", "wrong", "manual", "x"}


def read_bad_csv_keys(csv_path: Path) -> set[str]:
    """Read a review CSV and return image_key values marked for correction.

    Supported formats:
      1) Preferred: fix,image_key,... with fix = 1/yes/bad/x
      2) Simple one-column list: image_key
      3) Any CSV with image_key plus status = bad/fix/wrong/manual
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Bad-image CSV not found: {csv_path}")

    keys: set[str] = set()
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return keys
        fields = [name.strip() for name in reader.fieldnames]
        lower_to_name = {name.lower(): name for name in fields}
        key_col = lower_to_name.get("image_key") or lower_to_name.get("filename") or lower_to_name.get("file")
        fix_col = lower_to_name.get("fix") or lower_to_name.get("bad") or lower_to_name.get("manual") or lower_to_name.get("review")
        status_col = lower_to_name.get("status")

        for row in reader:
            if not key_col:
                continue
            key = str(row.get(key_col, "")).strip()
            if not key:
                continue

            # If there is no fix/status column, treat every listed image_key as bad.
            if not fix_col and not status_col:
                keys.add(key)
                continue

            should_fix = False
            if fix_col and truthy_text(row.get(fix_col, "")):
                should_fix = True
            if status_col and truthy_text(row.get(status_col, "")):
                should_fix = True
            if should_fix:
                keys.add(key)

    return keys


def write_bad_review_template(args: argparse.Namespace) -> None:
    """Create a CSV from auto_label_report.csv where the user marks fix=1 for bad images."""
    out = Path(args.out)
    report_path = Path(args.report) if getattr(args, "report", None) else out / "auto_label_report.csv"
    output_csv = Path(args.csv) if getattr(args, "csv", None) else out / "bad_review_list.csv"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing report CSV: {report_path}. Run autolabel first.")

    rows = []
    with report_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = str(row.get("status", "")).strip().lower()
            if bool(getattr(args, "suspects_only", False)) and status != "suspect":
                continue
            rows.append({
                "fix": "1" if (bool(getattr(args, "prefill_suspects", False)) and status == "suspect") else "",
                "image_key": row.get("image_key", ""),
                "class_name": row.get("class_name", ""),
                "rel_path": row.get("rel_path", ""),
                "count": row.get("count", ""),
                "mean_score": row.get("mean_score", ""),
                "status": row.get("status", ""),
                "review_image": row.get("review_image", ""),
                "notes": "",
            })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["fix", "image_key", "class_name", "rel_path", "count", "mean_score", "status", "review_image", "notes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote review CSV: {output_csv}")
    print("Open it in Excel. Put 1, yes, bad, x, or fix in the 'fix' column for images that need manual correction.")
    print("Then run:")
    print(f'python {Path(__file__).name} label --out "{out}" --bad_csv "{output_csv}"')


def safe_stem_from_key(image_key: str) -> str:
    stem = Path(image_key.replace("\\", "/")).with_suffix("").as_posix()
    return stem.replace("/", "__").replace(" ", "_")




# -----------------------------
# Automated pseudo-labeling from classical detector
# -----------------------------

def auto_resize_keep_aspect(image: np.ndarray, max_width: int) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    if w <= max_width:
        return image.copy(), 1.0
    scale = float(max_width) / float(w)
    return cv2.resize(image, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA), scale


def auto_find_blue_object_roi(
    image: np.ndarray,
    front_band: Tuple[float, float] = (0.18, 0.88),
) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
    """Find the blue 3D-printed object and return a front-face band ROI."""
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


def auto_preprocess_roi(roi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def auto_circle_score(edges: np.ndarray, gray: np.ndarray, x: float, y: float, r: float) -> float:
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
    return float(0.80 * edge_support + 0.20 * contrast)


def auto_hough_candidates(
    blurred: np.ndarray,
    edges: np.ndarray,
    min_radius: int,
    max_radius: int,
    hough_votes: int,
    multi_pass: bool,
) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    vote_values = [hough_votes]
    if multi_pass:
        vote_values = sorted(set([hough_votes - 6, hough_votes - 3, hough_votes, hough_votes + 4, hough_votes + 8]))
        vote_values = [v for v in vote_values if v >= 7]

    for votes in vote_values:
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(20, int(min_radius * 1.8)),
            param1=90,
            param2=float(votes),
            minRadius=int(min_radius),
            maxRadius=int(max_radius),
        )
        if circles is None:
            continue
        for cx, cy, radius in np.round(circles[0]).astype(int):
            score = auto_circle_score(edges, blurred, float(cx), float(cy), float(radius))
            candidates.append({"x": float(cx), "y": float(cy), "r": float(radius), "score": float(score), "method": f"hough_p2_{votes}"})
    return candidates


def auto_ellipse_candidates(
    edges: np.ndarray,
    gray: np.ndarray,
    min_radius: int,
    max_radius: int,
) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    processed_edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(processed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < (min_radius ** 2) * 0.15 or area > (max_radius ** 2) * 6.0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_radius or h < min_radius:
            continue
        if w > max_radius * 3 or h > max_radius * 3:
            continue
        aspect = w / float(max(h, 1))
        if not (0.45 <= aspect <= 2.5):
            continue
        if len(contour) < 5:
            continue
        (cx, cy), (axis_a, axis_b), _angle = cv2.fitEllipse(contour)
        major = max(axis_a, axis_b)
        minor = min(axis_a, axis_b)
        if major < min_radius or major > max_radius * 2.8:
            continue
        if minor < min_radius * 0.55:
            continue
        if major / max(minor, 1.0) > 3.0:
            continue
        radius = (major + minor) / 4.0
        score = auto_circle_score(edges, gray, cx, cy, radius) * 0.8
        candidates.append({"x": float(cx), "y": float(cy), "r": float(radius), "score": float(score), "method": "ellipse"})
    return candidates


def auto_nms(
    candidates: Sequence[Dict[str, object]],
    score_threshold: float,
    merge_factor: float,
    merge_min_distance: float,
) -> List[Dict[str, object]]:
    kept: List[Dict[str, object]] = []
    for cand in sorted(candidates, key=lambda d: float(d["score"]), reverse=True):
        if float(cand["score"]) < score_threshold:
            continue
        duplicate = False
        for ex in kept:
            dist = math.hypot(float(cand["x"]) - float(ex["x"]), float(cand["y"]) - float(ex["y"]))
            merge_distance = max(merge_min_distance, merge_factor * (float(cand["r"]) + float(ex["r"])))
            if dist < merge_distance:
                duplicate = True
                break
        if not duplicate:
            kept.append(dict(cand))
    return kept


def auto_filter_by_row(detections: Sequence[Dict[str, object]], row_tolerance: Optional[float]) -> List[Dict[str, object]]:
    dets = [dict(d) for d in detections]
    if len(dets) < 4:
        return dets
    ys = np.array([float(d["y"]) for d in dets], dtype=float)
    rs = np.array([float(d["r"]) for d in dets], dtype=float)
    median_y = float(np.median(ys))
    median_r = float(max(10.0, np.median(rs)))
    coarse_limit = max(60.0, 2.0 * median_r)
    coarse = [d for d in dets if abs(float(d["y"]) - median_y) <= coarse_limit]
    if len(coarse) < 4:
        coarse = dets
    xs2 = np.array([float(d["x"]) for d in coarse], dtype=float)
    ys2 = np.array([float(d["y"]) for d in coarse], dtype=float)
    try:
        slope, intercept = np.polyfit(xs2, ys2, 1)
    except Exception:
        return dets
    tolerance = row_tolerance if row_tolerance is not None else max(35.0, 1.35 * median_r)
    filtered = []
    for d in dets:
        predicted_y = slope * float(d["x"]) + intercept
        if abs(float(d["y"]) - predicted_y) <= tolerance:
            filtered.append(d)
    return filtered if len(filtered) >= 3 else dets


def auto_annotate(
    image: np.ndarray,
    detections_original: Sequence[Dict[str, object]],
    class_name: str,
    text: str,
) -> np.ndarray:
    out_img = image.copy()
    color = CLASS_COLORS.get(class_name, (0, 255, 255))
    for i, d in enumerate(sorted(detections_original, key=lambda x: float(x["x"])), start=1):
        cx = int(round(float(d["x"])))
        cy = int(round(float(d["y"])))
        r = max(2, int(round(float(d["r"]))))
        cv2.circle(out_img, (cx, cy), r, color, 2)
        cv2.circle(out_img, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(out_img, str(i), (cx + 4, max(15, cy - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.rectangle(out_img, (0, 0), (min(out_img.shape[1], 1500), 36), (0, 0, 0), -1)
    cv2.putText(out_img, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return out_img


def auto_detect_original_image(img: np.ndarray, args: argparse.Namespace) -> Tuple[List[Dict[str, object]], Tuple[int, int, int, int], Dict[str, np.ndarray], float]:
    resized, scale = auto_resize_keep_aspect(img, int(args.max_width))
    roi_xywh, blue_mask = auto_find_blue_object_roi(resized, front_band=(float(args.front_start), float(args.front_end)))
    x0, y0, w, h = roi_xywh
    x0 = max(0, min(resized.shape[1] - 1, x0))
    y0 = max(0, min(resized.shape[0] - 1, y0))
    w = max(1, min(resized.shape[1] - x0, w))
    h = max(1, min(resized.shape[0] - y0, h))
    roi_xywh = (x0, y0, w, h)
    roi = resized[y0:y0 + h, x0:x0 + w]
    enhanced, corrected, blurred, edges = auto_preprocess_roi(roi)
    candidates = auto_hough_candidates(blurred, edges, int(args.min_radius), int(args.max_radius), int(args.hough_votes), bool(args.multi_pass))
    if bool(args.use_ellipse_fallback):
        candidates.extend(auto_ellipse_candidates(edges, blurred, int(args.min_radius), int(args.max_radius)))
    dets = auto_nms(candidates, float(args.score_threshold), float(args.merge_factor), float(args.merge_min_distance))
    if not bool(args.no_row_filter):
        dets = auto_filter_by_row(dets, args.row_tolerance)
    dets = sorted(dets, key=lambda d: float(d["x"]))

    # Convert from ROI coordinates in resized image back to original image coordinates.
    dets_original = []
    for d in dets:
        dets_original.append({
            "x": (x0 + float(d["x"])) / scale,
            "y": (y0 + float(d["y"])) / scale,
            "r": float(d["r"]) / scale,
            "score": float(d["score"]),
            "method": str(d["method"]),
        })
    debug = {"blue_mask": blue_mask, "roi": roi, "enhanced": enhanced, "corrected": corrected, "edges": edges}
    return dets_original, roi_xywh, debug, scale


def auto_label_images(args: argparse.Namespace) -> None:
    """Create labels.csv automatically using a classical detector, then save review images for quick manual checking."""
    out = Path(args.out)
    ensure_dirs(out)

    provided = parse_class_dirs(args.class_dir)
    if provided:
        existing: Dict[str, Path] = {}
        src_json = out / "sources.json"
        if src_json.exists():
            try:
                existing = load_sources(out)
            except Exception:
                existing = {}
        existing.update(provided)
        save_sources(out, existing)

    class_dirs = load_sources(out)
    records = collect_image_records(class_dirs)
    if not records:
        raise FileNotFoundError("No images found in the provided class folders.")

    # Optional filters so you can fix only bad auto-labeled images instead of opening every image.
    only_items = getattr(args, "only", None)
    if only_items:
        needles = [str(x).lower() for x in only_items]
        records = [r for r in records if any(n in str(r["image_key"]).lower() or n in str(r["rel_path"]).lower() for n in needles)]
        if not records:
            raise FileNotFoundError(f"No images matched --only filters: {only_items}")

    if bool(getattr(args, "suspects_only", False)):
        report_path = out / "auto_label_report.csv"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing {report_path}. Run autolabel first, or use --only instead.")
        suspect_keys = set()
        with report_path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status", "").lower() == "suspect":
                    suspect_keys.add(row.get("image_key", ""))
        records = [r for r in records if str(r["image_key"]) in suspect_keys]
        if not records:
            print("No suspect images found in auto_label_report.csv.")
            return

    label_csv = out / "labels.csv"
    labels = load_labels(label_csv)
    review_dir = out / "auto_label_review"
    debug_dir = out / "auto_label_debug"
    review_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.debug):
        debug_dir.mkdir(parents=True, exist_ok=True)

    report_rows: List[Dict[str, object]] = []
    n_processed = 0
    n_skipped = 0
    n_suspect = 0

    print(f"Auto-labeling {len(records)} images...")
    for rec in tqdm(records):
        image_key = str(rec["image_key"])
        class_name = str(rec["class_name"])
        rel_path = str(rec["rel_path"])
        path = Path(rec["path"])

        if image_key in labels and not bool(args.overwrite_existing):
            n_skipped += 1
            continue

        img = cv2.imread(str(path))
        if img is None:
            report_rows.append({"image_key": image_key, "class_name": class_name, "rel_path": rel_path, "count": 0, "mean_score": 0.0, "status": "unreadable"})
            continue

        dets, roi_xywh, debug, scale = auto_detect_original_image(img, args)
        points = [(float(d["x"]), float(d["y"]), float(d["r"])) for d in dets]
        labels[image_key] = {"class_name": class_name, "rel_path": rel_path, "points": points}

        count = len(dets)
        mean_score = float(np.mean([float(d["score"]) for d in dets])) if dets else 0.0
        low_score = mean_score < float(args.suspect_score)
        count_bad = count < int(args.min_count) or count > int(args.max_count)
        status = "suspect" if (low_score or count_bad) else "ok"
        if status == "suspect":
            n_suspect += 1

        review_class_dir = review_dir / class_name
        review_class_dir.mkdir(parents=True, exist_ok=True)
        stem = safe_stem_from_key(image_key)
        text = f"{status.upper()} class={class_name} count={count} mean_score={mean_score:.3f}  {rel_path}"
        ann = auto_annotate(img, dets, class_name, text)
        cv2.imwrite(str(review_class_dir / f"{stem}__auto.png"), ann)

        if bool(args.debug):
            ddir = debug_dir / class_name
            ddir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(ddir / f"{stem}__blue_mask.png"), debug["blue_mask"])
            cv2.imwrite(str(ddir / f"{stem}__roi.png"), debug["roi"])
            cv2.imwrite(str(ddir / f"{stem}__corrected.png"), debug["corrected"])
            cv2.imwrite(str(ddir / f"{stem}__edges.png"), debug["edges"])

        report_rows.append({
            "image_key": image_key,
            "class_name": class_name,
            "rel_path": rel_path,
            "count": count,
            "mean_score": f"{mean_score:.4f}",
            "status": status,
            "review_image": str((review_class_dir / f"{stem}__auto.png").relative_to(out)),
        })
        n_processed += 1

    save_labels(label_csv, labels)
    report_path = out / "auto_label_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["image_key", "class_name", "rel_path", "count", "mean_score", "status", "review_image"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    print("\nAuto-label finished.")
    print(f"Labels saved:        {label_csv}")
    print(f"Review overlays:     {review_dir}")
    print(f"Report CSV:          {report_path}")
    print(f"Processed:           {n_processed}")
    print(f"Skipped existing:    {n_skipped}")
    print(f"Suspect images:      {n_suspect}")
    print("\nNext step: open the auto_label_review folder. If an image is wrong, run the normal 'label' command and fix only those images.")

# -----------------------------
# Manual labeling UI
# -----------------------------

def label_images(args: argparse.Namespace) -> None:
    out = Path(args.out)
    ensure_dirs(out)

    provided = parse_class_dirs(args.class_dir)
    if provided:
        # Save/update source folders when provided.
        existing: Dict[str, Path] = {}
        src_json = out / "sources.json"
        if src_json.exists():
            try:
                existing = load_sources(out)
            except Exception:
                existing = {}
        existing.update(provided)
        save_sources(out, existing)

    class_dirs = load_sources(out)
    missing = [name for name in SHAPE_CLASSES if name not in class_dirs]
    if missing:
        print(f"Warning: missing folders for classes: {', '.join(missing)}")

    records = collect_image_records(class_dirs)
    if not records:
        raise FileNotFoundError("No images found in the provided class folders.")

    # Optional filters so you only open the bad/wrong auto-labeled images.
    only_items = getattr(args, "only", None)
    if only_items:
        needles = [str(x).lower() for x in only_items]
        records = [r for r in records if any(n in str(r["image_key"]).lower() or n in str(r["rel_path"]).lower() for n in needles)]
        if not records:
            raise FileNotFoundError(f"No images matched --only filters: {only_items}")

    if bool(getattr(args, "suspects_only", False)):
        report_path = out / "auto_label_report.csv"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing {report_path}. Run autolabel first, or use --only/--bad_csv instead.")
        suspect_keys = set()
        with report_path.open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("status", "").lower() == "suspect":
                    suspect_keys.add(row.get("image_key", ""))
        records = [r for r in records if str(r["image_key"]) in suspect_keys]
        if not records:
            print("No suspect images found in auto_label_report.csv.")
            return

    bad_csv = getattr(args, "bad_csv", None)
    if bad_csv:
        bad_keys = read_bad_csv_keys(Path(bad_csv))
        if not bad_keys:
            print(f"No images marked for correction in {bad_csv}.")
            return
        # Match by exact image_key first. Also allow filename/rel_path text in case the CSV is simple.
        bad_needles = {k.lower() for k in bad_keys}
        records = [
            r for r in records
            if str(r["image_key"]) in bad_keys
            or str(r["rel_path"]) in bad_keys
            or Path(str(r["rel_path"])).name in bad_keys
            or any(k in str(r["image_key"]).lower() or k in str(r["rel_path"]).lower() for k in bad_needles)
        ]
        if not records:
            raise FileNotFoundError(f"No images from --bad_csv matched your source folders: {bad_csv}")

    print(f"Manual labeler will open {len(records)} image(s).")

    label_csv = out / "labels.csv"
    labels = load_labels(label_csv)
    current_radius = float(args.radius)
    max_display_w = int(args.max_display_w)
    max_display_h = int(args.max_display_h)
    win = "Multi-class rivet labeler"

    idx = int(args.start_index)
    idx = max(0, min(idx, len(records) - 1))

    while 0 <= idx < len(records):
        rec = records[idx]
        img_path = Path(rec["path"])
        image_key = str(rec["image_key"])
        class_name = str(rec["class_name"])
        rel_path = str(rec["rel_path"])

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not read {img_path}; skipping.")
            idx += 1
            continue
        h, w = img.shape[:2]
        points = list(labels.get(image_key, {}).get("points", []))
        undo_stack: List[List[Tuple[float, float, float]]] = []

        scale = min(max_display_w / w, max_display_h / h, 1.0)
        disp_w, disp_h = int(w * scale), int(h * scale)
        color = CLASS_COLORS.get(class_name, (0, 255, 255))

        def redraw() -> np.ndarray:
            view = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            for k, (x, y, r) in enumerate(points, start=1):
                sx, sy, sr = int(round(x * scale)), int(round(y * scale)), int(round(r * scale))
                cv2.circle(view, (sx, sy), max(2, sr), color, 2)
                cv2.circle(view, (sx, sy), 3, (0, 0, 255), -1)
                cv2.putText(view, str(k), (sx + 4, sy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            status = (
                f"{idx+1}/{len(records)}  class={class_name}  points={len(points)}  "
                f"radius={current_radius:.1f}px  {rel_path}"
            )
            cv2.rectangle(view, (0, 0), (min(disp_w, 1450), 38), (0, 0, 0), -1)
            cv2.putText(view, status, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
            help_text = "Left click add | Right click delete | +/- radius | u undo | c clear | n next | p previous | s save | q quit"
            cv2.rectangle(view, (0, disp_h - 32), (min(disp_w, 1450), disp_h), (0, 0, 0), -1)
            cv2.putText(view, help_text, (8, disp_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            return view

        def mouse_cb(event, mx, my, flags, param):
            nonlocal points
            ox, oy = mx / scale, my / scale
            if event == cv2.EVENT_LBUTTONDOWN:
                undo_stack.append(list(points))
                points.append((float(ox), float(oy), float(current_radius)))
                cv2.imshow(win, redraw())
            elif event == cv2.EVENT_RBUTTONDOWN:
                if points:
                    dists = [(i, (x - ox) ** 2 + (y - oy) ** 2) for i, (x, y, r) in enumerate(points)]
                    nearest_i, nearest_d = min(dists, key=lambda t: t[1])
                    if nearest_d <= (max(current_radius, 20.0) * 2.5) ** 2:
                        undo_stack.append(list(points))
                        points.pop(nearest_i)
                cv2.imshow(win, redraw())

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, mouse_cb)
        cv2.imshow(win, redraw())

        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == 255:
                continue
            if key in [ord("+"), ord("=")]:
                current_radius += 2.0
                cv2.imshow(win, redraw())
            elif key in [ord("-"), ord("_")]:
                current_radius = max(2.0, current_radius - 2.0)
                cv2.imshow(win, redraw())
            elif key == ord("u"):
                if undo_stack:
                    points = undo_stack.pop()
                    cv2.imshow(win, redraw())
            elif key == ord("c"):
                undo_stack.append(list(points))
                points = []
                cv2.imshow(win, redraw())
            elif key == ord("s"):
                labels[image_key] = {"class_name": class_name, "rel_path": rel_path, "points": points}
                save_labels(label_csv, labels)
                print(f"Saved {len(points)} points for {image_key}")
            elif key == ord("n"):
                labels[image_key] = {"class_name": class_name, "rel_path": rel_path, "points": points}
                save_labels(label_csv, labels)
                idx += 1
                break
            elif key == ord("p"):
                labels[image_key] = {"class_name": class_name, "rel_path": rel_path, "points": points}
                save_labels(label_csv, labels)
                idx = max(0, idx - 1)
                break
            elif key == ord("q") or key == 27:
                labels[image_key] = {"class_name": class_name, "rel_path": rel_path, "points": points}
                save_labels(label_csv, labels)
                cv2.destroyWindow(win)
                print(f"Saved labels to {label_csv}")
                return

    cv2.destroyAllWindows()
    save_labels(label_csv, labels)
    print(f"Saved labels to {label_csv}")


# -----------------------------
# Masks, heatmaps, augmentation
# -----------------------------

def draw_original_class_mask(shape_hw: Tuple[int, int], centers_r: Sequence[Tuple[float, float, float]], class_id: int) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    for x, y, r in centers_r:
        cv2.circle(mask, (int(round(x)), int(round(y))), max(1, int(round(r))), int(class_id), -1)
    return mask


def filter_visible_center_objs(center_objs: Sequence[Dict[str, object]], w: int, h: int) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for c in center_objs:
        x, y = float(c["x"]), float(c["y"])
        if 0 <= x < w and 0 <= y < h:
            out.append(dict(c))
    return out


def resize_image_mask_centers(
    img: np.ndarray,
    mask: np.ndarray,
    centers_r: Sequence[Tuple[float, float, float]],
    class_name: str,
    target_w: int = IMG_W,
    target_h: int = IMG_H,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    h, w = img.shape[:2]
    sx, sy = target_w / w, target_h / h
    img_r = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    mask_r = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    center_objs = [
        {"x": float(x * sx), "y": float(y * sy), "class_name": class_name, "class_id": int(CLASS_TO_ID[class_name])}
        for x, y, _ in centers_r
    ]
    center_objs = filter_visible_center_objs(center_objs, target_w, target_h)
    return img_r, mask_r, center_objs


def transform_points_affine(points: Sequence[Tuple[float, float]], M: np.ndarray) -> List[Tuple[float, float]]:
    if not points:
        return []
    pts = np.asarray([[x, y, 1.0] for x, y in points], dtype=np.float32)
    out = pts @ M.T
    return [(float(x), float(y)) for x, y in out]


def transform_points_perspective(points: Sequence[Tuple[float, float]], H: np.ndarray) -> List[Tuple[float, float]]:
    if not points:
        return []
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in out]


def transform_center_objs_affine(center_objs: Sequence[Dict[str, object]], M: np.ndarray) -> List[Dict[str, object]]:
    pts = [(float(c["x"]), float(c["y"])) for c in center_objs]
    pts_t = transform_points_affine(pts, M)
    out = []
    for c, (x, y) in zip(center_objs, pts_t):
        item = dict(c)
        item["x"] = float(x)
        item["y"] = float(y)
        out.append(item)
    return out


def transform_center_objs_perspective(center_objs: Sequence[Dict[str, object]], H: np.ndarray) -> List[Dict[str, object]]:
    pts = [(float(c["x"]), float(c["y"])) for c in center_objs]
    pts_t = transform_points_perspective(pts, H)
    out = []
    for c, (x, y) in zip(center_objs, pts_t):
        item = dict(c)
        item["x"] = float(x)
        item["y"] = float(y)
        out.append(item)
    return out


def random_motion_blur(img: np.ndarray, max_kernel: int = 7) -> np.ndarray:
    k = random.choice([3, 5, 7])
    k = min(k, max_kernel)
    kernel = np.zeros((k, k), dtype=np.float32)
    if random.random() < 0.5:
        kernel[k // 2, :] = 1.0
    else:
        kernel[:, k // 2] = 1.0
    kernel /= k
    return cv2.filter2D(img, -1, kernel)


def apply_appearance_aug(img: np.ndarray) -> np.ndarray:
    out = img.astype(np.float32)

    if random.random() < 0.75:
        alpha = random.uniform(0.75, 1.25)
        beta = random.uniform(-24, 24)
        out = out * alpha + beta

    out = np.clip(out, 0, 255).astype(np.uint8)

    if random.random() < 0.45:
        gamma = random.uniform(0.70, 1.40)
        table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
        out = cv2.LUT(out, table)

    if random.random() < 0.35:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-4, 4)) % 180
        hsv[..., 1] *= random.uniform(0.85, 1.15)
        hsv[..., 2] *= random.uniform(0.85, 1.15)
        hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if random.random() < 0.20:
        out = cv2.GaussianBlur(out, (5, 5), 0)
    if random.random() < 0.18:
        out = random_motion_blur(out, 7)
    if random.random() < 0.35:
        sigma = random.uniform(2.0, 10.0)
        noise = np.random.normal(0.0, sigma, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.25:
        q = random.randint(55, 95)
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return out


def augment_one(
    img: np.ndarray,
    mask: np.ndarray,
    center_objs: Sequence[Dict[str, object]],
    w: int = IMG_W,
    h: int = IMG_H,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    img_aug = img.copy()
    mask_aug = mask.copy()
    centers_aug = [dict(c) for c in center_objs]

    if random.random() < 0.90:
        angle = random.uniform(-9.0, 9.0)
        scale = random.uniform(0.88, 1.14)
        tx = random.uniform(-0.065, 0.065) * w
        ty = random.uniform(-0.065, 0.065) * h
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        img_aug = cv2.warpAffine(img_aug, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask_aug = cv2.warpAffine(mask_aug, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        centers_aug = transform_center_objs_affine(centers_aug, M)

    if random.random() < 0.35:
        max_jitter = random.uniform(4.0, 18.0)
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        jitter = np.random.uniform(-max_jitter, max_jitter, size=(4, 2)).astype(np.float32)
        dst = src + jitter
        Hm = cv2.getPerspectiveTransform(src, dst)
        img_aug = cv2.warpPerspective(img_aug, Hm, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask_aug = cv2.warpPerspective(mask_aug, Hm, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        centers_aug = transform_center_objs_perspective(centers_aug, Hm)

    centers_aug = filter_visible_center_objs(centers_aug, w, h)
    mask_aug = np.clip(mask_aug, 0, NUM_FG_CLASSES).astype(np.uint8)
    img_aug = apply_appearance_aug(img_aug)
    return img_aug, mask_aug, centers_aug


def make_class_heatmaps(h: int, w: int, center_objs: Sequence[Dict[str, object]], sigma: float = 4.0) -> np.ndarray:
    heat = np.zeros((NUM_FG_CLASSES, h, w), dtype=np.float32)
    radius = int(max(3, sigma * 3))
    for c in center_objs:
        class_id = int(c["class_id"])
        ch = class_id - 1
        x, y = float(c["x"]), float(c["y"])
        cx, cy = int(round(x)), int(round(y))
        x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)[:, None]
        patch = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma * sigma))
        heat[ch, y0:y1, x0:x1] = np.maximum(heat[ch, y0:y1, x0:x1], patch)
    return heat


def build_split_map_stratified(labeled_items: List[Dict[str, object]], out: Path, val_frac: float, test_frac: float) -> Dict[str, str]:
    split_csv = out / "splits.csv"
    existing = read_split_csv(split_csv)
    if existing:
        return existing

    rng = random.Random(RANDOM_SEED)
    split_map: Dict[str, str] = {}
    by_class: Dict[str, List[str]] = {c: [] for c in SHAPE_CLASSES}
    for item in labeled_items:
        by_class[str(item["class_name"])].append(str(item["image_key"]))

    for class_name, keys in by_class.items():
        if not keys:
            continue
        rng.shuffle(keys)
        n = len(keys)
        # If small class count, still try to place at least one in val/test when possible.
        n_test = max(1, int(round(n * test_frac))) if n >= 3 else 0
        n_val = max(1, int(round(n * val_frac))) if n >= 3 else 0
        # Keep at least one train sample.
        if n_test + n_val >= n:
            n_test = 1 if n >= 3 else 0
            n_val = 1 if n >= 3 else 0
        for i, key in enumerate(keys):
            if i < n_test:
                split_map[key] = "test"
            elif i < n_test + n_val:
                split_map[key] = "val"
            else:
                split_map[key] = "train"

    write_split_csv(split_csv, split_map)
    return split_map


def save_sample(
    out: Path,
    split: str,
    sample_id: str,
    img: np.ndarray,
    class_mask: np.ndarray,
    center_objs: Sequence[Dict[str, object]],
    source_image_key: str,
    source_class: str,
    rows: List[Dict[str, str]],
) -> None:
    img_path = out / "dataset" / "images" / split / f"{sample_id}.jpg"
    mask_path = out / "dataset" / "masks" / split / f"{sample_id}.png"
    label_path = out / "dataset" / "labels" / split / f"{sample_id}.json"
    cv2.imwrite(str(img_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imwrite(str(mask_path), class_mask.astype(np.uint8))
    with label_path.open("w", encoding="utf-8") as f:
        json.dump({
            "image": f"{sample_id}.jpg",
            "source_image_key": source_image_key,
            "source_class": source_class,
            "centers": [
                {
                    "x": float(c["x"]),
                    "y": float(c["y"]),
                    "class_name": str(c["class_name"]),
                    "class_id": int(c["class_id"]),
                }
                for c in center_objs
            ],
        }, f, indent=2)
    rows.append({
        "split": split,
        "sample_id": sample_id,
        "image": str(img_path.relative_to(out)),
        "mask": str(mask_path.relative_to(out)),
        "label": str(label_path.relative_to(out)),
        "source_image_key": source_image_key,
        "source_class": source_class,
        "n_centers": str(len(center_objs)),
    })


def build_dataset(args: argparse.Namespace) -> None:
    out = Path(args.out)
    ensure_dirs(out)
    class_dirs = load_sources(out)
    records = collect_image_records(class_dirs)
    rec_by_key = {str(r["image_key"]): r for r in records}

    label_csv = out / "labels.csv"
    labels = load_labels(label_csv)
    if not labels:
        raise RuntimeError(f"No labels found. Run label first. Expected {label_csv}")

    labeled_items: List[Dict[str, object]] = []
    for key, item in labels.items():
        pts = item.get("points", [])
        if key in rec_by_key and len(pts) > 0:
            labeled_items.append({
                "image_key": key,
                "class_name": str(item["class_name"]),
                "rel_path": str(item.get("rel_path", key.split("/", 1)[-1])),
                "points": pts,
                "path": rec_by_key[key]["path"],
            })
    if not labeled_items:
        raise RuntimeError("No labeled images matched files in the configured source folders.")

    if args.overwrite:
        ds = out / "dataset"
        if ds.exists():
            shutil.rmtree(ds)
        ensure_dirs(out)

    split_map = build_split_map_stratified(labeled_items, out, float(args.val_frac), float(args.test_frac))
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    rng = random.Random(RANDOM_SEED)

    rows: List[Dict[str, str]] = []
    originals_by_split: Dict[str, List[Tuple[str, str, np.ndarray, np.ndarray, List[Dict[str, object]]]]] = {
        "train": [], "val": [], "test": []
    }

    print("Creating original resized samples...")
    for item in tqdm(labeled_items):
        image_key = str(item["image_key"])
        class_name = str(item["class_name"])
        class_id = CLASS_TO_ID[class_name]
        split = split_map.get(image_key, "train")
        img = cv2.imread(str(item["path"]))
        if img is None:
            print(f"Could not read: {item['path']}")
            continue
        centers_r = list(item["points"])
        class_mask = draw_original_class_mask(img.shape[:2], centers_r, class_id)
        img_r, mask_r, center_objs = resize_image_mask_centers(img, class_mask, centers_r, class_name, IMG_W, IMG_H)
        sample_id = safe_stem_from_key(image_key) + "__orig"
        save_sample(out, split, sample_id, img_r, mask_r, center_objs, image_key, class_name, rows)
        originals_by_split[split].append((image_key, class_name, img_r, mask_r, center_objs))

    train_items = originals_by_split["train"]
    if not train_items:
        raise RuntimeError("No train items found after split. Check splits.csv or label more images.")

    target_train = int(args.target_train)
    current_train = len(train_items)
    if bool(args.balance_classes):
        by_class: Dict[str, List[Tuple[str, str, np.ndarray, np.ndarray, List[Dict[str, object]]]]] = {c: [] for c in SHAPE_CLASSES}
        for item in train_items:
            by_class[item[1]].append(item)
        present_classes = [c for c in SHAPE_CLASSES if by_class[c]]
        if not present_classes:
            raise RuntimeError("No present classes in train split.")
        base_target = target_train // len(present_classes)
        remainder = target_train % len(present_classes)
        target_by_class = {c: base_target + (1 if i < remainder else 0) for i, c in enumerate(present_classes)}
        print("Train originals by class:")
        for c in present_classes:
            print(f"  {c}: originals={len(by_class[c])}, target_train_samples={target_by_class[c]}")
        aug_counter = 0
        for c in present_classes:
            current = len(by_class[c])
            needed = max(0, target_by_class[c] - current)
            print(f"Adding {needed} augmented train samples for class={c}")
            for _ in tqdm(range(needed), desc=f"augment {c}"):
                source_key, class_name, img_r, mask_r, center_objs = rng.choice(by_class[c])
                img_a, mask_a, centers_a = augment_one(img_r, mask_r, center_objs, IMG_W, IMG_H)
                if len(centers_a) == 0 or np.count_nonzero(mask_a) < 10:
                    img_a, mask_a, centers_a = img_r.copy(), mask_r.copy(), [dict(x) for x in center_objs]
                sample_id = f"{safe_stem_from_key(source_key)}__aug_{aug_counter:05d}"
                aug_counter += 1
                save_sample(out, "train", sample_id, img_a, mask_a, centers_a, source_key, class_name, rows)
    else:
        needed = max(0, target_train - current_train)
        print(f"Train originals: {current_train}. Adding {needed} augmented train samples to reach {target_train} total.")
        for aug_idx in tqdm(range(needed)):
            source_key, class_name, img_r, mask_r, center_objs = rng.choice(train_items)
            img_a, mask_a, centers_a = augment_one(img_r, mask_r, center_objs, IMG_W, IMG_H)
            if len(centers_a) == 0 or np.count_nonzero(mask_a) < 10:
                img_a, mask_a, centers_a = img_r.copy(), mask_r.copy(), [dict(x) for x in center_objs]
            sample_id = f"{safe_stem_from_key(source_key)}__aug_{aug_idx:05d}"
            save_sample(out, "train", sample_id, img_a, mask_a, centers_a, source_key, class_name, rows)

    meta_path = out / "dataset" / "metadata.csv"
    with meta_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["split", "sample_id", "image", "mask", "label", "source_image_key", "source_class", "n_centers"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Dataset saved to: {out / 'dataset'}")
    print(f"Metadata: {meta_path}")
    print("Counts:")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {sum(1 for r in rows if r['split'] == split)}")
    print("Counts by split/class:")
    for split in ["train", "val", "test"]:
        msg = []
        for c in SHAPE_CLASSES:
            msg.append(f"{c}={sum(1 for r in rows if r['split'] == split and r['source_class'] == c)}")
        print(f"  {split}: " + ", ".join(msg))


# -----------------------------
# PyTorch dataset and model
# -----------------------------

class RivetShapeDataset(Dataset):
    def __init__(self, project_dir: Path, split: str, heat_sigma: float = 4.0):
        self.project_dir = project_dir
        self.split = split
        self.heat_sigma = heat_sigma
        meta = project_dir / "dataset" / "metadata.csv"
        if not meta.exists():
            raise FileNotFoundError(f"Missing {meta}. Run build first.")
        rows = []
        with meta.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    rows.append(row)
        if not rows:
            raise RuntimeError(f"No rows for split={split}")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img_path = self.project_dir / row["image"]
        mask_path = self.project_dir / row["mask"]
        label_path = self.project_dir / row["label"]

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(str(img_path))
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        class_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if class_mask is None:
            raise FileNotFoundError(str(mask_path))
        class_mask = np.clip(class_mask.astype(np.int64), 0, NUM_FG_CLASSES)

        with label_path.open("r", encoding="utf-8") as f:
            lab = json.load(f)
        center_objs = lab.get("centers", [])
        heat = make_class_heatmaps(class_mask.shape[0], class_mask.shape[1], center_objs, sigma=self.heat_sigma)

        x = torch.from_numpy(img.transpose(2, 0, 1)).float()
        y_mask = torch.from_numpy(class_mask).long()
        y_heat = torch.from_numpy(heat).float()
        return x, y_mask, y_heat, row["sample_id"]


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SmallUNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = NUM_SEG_CLASSES + NUM_FG_CLASSES, base: int = 32):
        super().__init__()
        self.c1 = ConvBlock(in_ch, base)
        self.c2 = ConvBlock(base, base * 2)
        self.c3 = ConvBlock(base * 2, base * 4)
        self.c4 = ConvBlock(base * 4, base * 8)
        self.bot = ConvBlock(base * 8, base * 16)
        self.pool = nn.MaxPool2d(2)
        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.d4 = ConvBlock(base * 16, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = ConvBlock(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = ConvBlock(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(self.pool(c1))
        c3 = self.c3(self.pool(c2))
        c4 = self.c4(self.pool(c3))
        b = self.bot(self.pool(c4))
        x = self.u4(b)
        x = self.d4(torch.cat([x, c4], dim=1))
        x = self.u3(x)
        x = self.d3(torch.cat([x, c3], dim=1))
        x = self.u2(x)
        x = self.d2(torch.cat([x, c2], dim=1))
        x = self.u1(x)
        x = self.d1(torch.cat([x, c1], dim=1))
        return self.out(x)


def multiclass_dice_loss(seg_logits, mask_target, eps: float = 1e-6, include_background: bool = False):
    probs = torch.softmax(seg_logits, dim=1)
    onehot = F.one_hot(mask_target, num_classes=NUM_SEG_CLASSES).permute(0, 3, 1, 2).float()
    if not include_background:
        probs = probs[:, 1:, :, :]
        onehot = onehot[:, 1:, :, :]
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dim=dims)
    denom = probs.sum(dim=dims) + onehot.sum(dim=dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def split_logits(logits):
    seg_logits = logits[:, :NUM_SEG_CLASSES, :, :]
    heat_logits = logits[:, NUM_SEG_CLASSES:, :, :]
    return seg_logits, heat_logits


def make_ce_loss(device, bg_weight: float = 0.20):
    weights = torch.ones(NUM_SEG_CLASSES, dtype=torch.float32, device=device)
    weights[0] = float(bg_weight)
    return nn.CrossEntropyLoss(weight=weights)


def train_one_epoch(model, loader, optimizer, device, bg_weight: float):
    model.train()
    total = 0.0
    n = 0
    ce_loss = make_ce_loss(device, bg_weight)
    for imgs, masks, heats, _ in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        heats = heats.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        seg_logits, heat_logits = split_logits(logits)
        loss_seg = ce_loss(seg_logits, masks) + multiclass_dice_loss(seg_logits, masks)
        loss_heat = F.mse_loss(torch.sigmoid(heat_logits), heats)
        loss = loss_seg + 1.0 * loss_heat
        loss.backward()
        optimizer.step()
        bs = imgs.size(0)
        total += float(loss.item()) * bs
        n += bs
    return total / max(1, n)


@torch.no_grad()
def validate_loss(model, loader, device, bg_weight: float):
    model.eval()
    total = 0.0
    n = 0
    ce_loss = make_ce_loss(device, bg_weight)
    for imgs, masks, heats, _ in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        heats = heats.to(device)
        logits = model(imgs)
        seg_logits, heat_logits = split_logits(logits)
        loss_seg = ce_loss(seg_logits, masks) + multiclass_dice_loss(seg_logits, masks)
        loss_heat = F.mse_loss(torch.sigmoid(heat_logits), heats)
        loss = loss_seg + 1.0 * loss_heat
        bs = imgs.size(0)
        total += float(loss.item()) * bs
        n += bs
    return total / max(1, n)


def train_model(args: argparse.Namespace) -> None:
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Run: pip install torch")
    out = Path(args.out)
    ensure_dirs(out)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    train_ds = RivetShapeDataset(out, "train", heat_sigma=float(args.heat_sigma))
    val_ds = RivetShapeDataset(out, "val", heat_sigma=float(args.heat_sigma))
    train_loader = DataLoader(train_ds, batch_size=int(args.batch), shuffle=True, num_workers=int(args.workers), pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=int(args.batch), shuffle=False, num_workers=int(args.workers), pin_memory=(device.type == "cuda"))

    model = SmallUNet(base=int(args.base)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)

    best_val = float("inf")
    ckpt_path = out / "checkpoints" / "best_unet_shapes.pt"
    history_path = out / "reports" / "train_history_shapes.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        writer.writeheader()

        patience = int(args.patience)
        bad = 0
        for epoch in range(1, int(args.epochs) + 1):
            tr = train_one_epoch(model, train_loader, optimizer, device, bg_weight=float(args.bg_weight))
            va = validate_loss(model, val_loader, device, bg_weight=float(args.bg_weight))
            writer.writerow({"epoch": epoch, "train_loss": f"{tr:.6f}", "val_loss": f"{va:.6f}", "lr": optimizer.param_groups[0]["lr"]})
            f.flush()
            print(f"Epoch {epoch:03d}: train_loss={tr:.5f}  val_loss={va:.5f}")

            if va < best_val:
                best_val = va
                bad = 0
                torch.save({
                    "model_state": model.state_dict(),
                    "base": int(args.base),
                    "img_w": IMG_W,
                    "img_h": IMG_H,
                    "heat_sigma": float(args.heat_sigma),
                    "classes": SHAPE_CLASSES,
                    "num_seg_classes": NUM_SEG_CLASSES,
                    "num_fg_classes": NUM_FG_CLASSES,
                    "best_val_loss": best_val,
                }, ckpt_path)
                print(f"  saved best checkpoint: {ckpt_path}")
            else:
                bad += 1
                if bad >= patience:
                    print(f"Early stopping after {patience} epochs without improvement.")
                    break

    print(f"Best val loss: {best_val:.6f}")
    print(f"Checkpoint: {ckpt_path}")


# -----------------------------
# Evaluation and center extraction
# -----------------------------

def load_model(project_dir: Path, cpu: bool = False):
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Run: pip install torch")
    ckpt_path = project_dir / "checkpoints" / "best_unet_shapes.pt"
    if not ckpt_path.exists():
        # Backward-friendly error if they accidentally use the older script checkpoint.
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}. Run the train command for the multi-class script.")
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = SmallUNet(base=int(ckpt.get("base", 32))).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device, ckpt


def extract_local_maxima(
    heat: np.ndarray,
    gate_prob: Optional[np.ndarray] = None,
    heat_thresh: float = 0.30,
    mask_thresh: float = 0.30,
    min_dist: int = 8,
    max_centers: int = 200,
) -> List[Tuple[float, float, float]]:
    hm = heat.astype(np.float32).copy()
    if gate_prob is not None:
        hm = hm * (gate_prob > mask_thresh).astype(np.float32)
    k = max(3, int(min_dist) * 2 + 1)
    kernel = np.ones((k, k), dtype=np.uint8)
    dil = cv2.dilate(hm, kernel)
    ys, xs = np.where((hm >= heat_thresh) & (hm >= dil - 1e-6))
    candidates = [(float(hm[y, x]), float(x), float(y)) for y, x in zip(ys, xs)]
    candidates.sort(reverse=True, key=lambda t: t[0])
    selected: List[Tuple[float, float, float]] = []
    min_d2 = float(min_dist * min_dist)
    for conf, x, y in candidates:
        if all((x - sx) ** 2 + (y - sy) ** 2 >= min_d2 for _, sx, sy in selected):
            selected.append((conf, x, y))
            if len(selected) >= max_centers:
                break
    return [(x, y, conf) for conf, x, y in selected]


def extract_centers_multiclass(
    seg_probs: np.ndarray,
    heat_probs: np.ndarray,
    heat_thresh: float = 0.30,
    mask_thresh: float = 0.40,
    min_dist: int = 8,
) -> List[Dict[str, object]]:
    preds: List[Dict[str, object]] = []
    for class_name in SHAPE_CLASSES:
        class_id = CLASS_TO_ID[class_name]
        ch = class_id - 1
        centers = extract_local_maxima(
            heat_probs[ch],
            gate_prob=seg_probs[class_id],
            heat_thresh=heat_thresh,
            mask_thresh=mask_thresh,
            min_dist=min_dist,
        )
        for x, y, conf in centers:
            preds.append({
                "x": float(x),
                "y": float(y),
                "conf": float(conf),
                "class_name": class_name,
                "class_id": class_id,
            })
    preds.sort(key=lambda p: float(p["conf"]), reverse=True)
    return preds


def match_centers_multiclass(preds: Sequence[Dict[str, object]], gts: Sequence[Dict[str, object]], match_dist: float) -> Dict[str, object]:
    per_class = {}
    total_tp = total_fp = total_fn = 0
    all_errors: List[float] = []

    for class_name in SHAPE_CLASSES:
        class_id = CLASS_TO_ID[class_name]
        p_cls = [p for p in preds if int(p["class_id"]) == class_id]
        g_cls = [g for g in gts if int(g["class_id"]) == class_id]
        used_gt = set()
        errors: List[float] = []
        tp = 0
        fp = 0
        for p in sorted(p_cls, key=lambda d: float(d.get("conf", 1.0)), reverse=True):
            px, py = float(p["x"]), float(p["y"])
            best_i = None
            best_d = float("inf")
            for i, g in enumerate(g_cls):
                if i in used_gt:
                    continue
                d = math.hypot(px - float(g["x"]), py - float(g["y"]))
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i is not None and best_d <= match_dist:
                used_gt.add(best_i)
                tp += 1
                errors.append(best_d)
            else:
                fp += 1
        fn = len(g_cls) - len(used_gt)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[class_name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "errors": errors,
            "mean_error": float(np.mean(errors)) if errors else float("nan"),
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_errors.extend(errors)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "errors": all_errors,
        "mean_error": float(np.mean(all_errors)) if all_errors else float("nan"),
        "median_error": float(np.median(all_errors)) if all_errors else float("nan"),
        "p95_error": float(np.percentile(all_errors, 95)) if all_errors else float("nan"),
        "per_class": per_class,
    }


def class_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, class_id: int) -> float:
    pred = pred_mask == class_id
    gt = gt_mask == class_id
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return float("nan")
    inter = np.logical_and(pred, gt).sum()
    return float(inter / union)


@torch.no_grad()
def predict_one(model, device, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    img_r = cv2.resize(img_bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    logits = model(x)
    seg_logits, heat_logits = split_logits(logits)
    seg_probs = torch.softmax(seg_logits, dim=1)[0].detach().cpu().numpy()
    heat_probs = torch.sigmoid(heat_logits)[0].detach().cpu().numpy()
    return seg_probs, heat_probs


def evaluate_model(args: argparse.Namespace) -> None:
    out = Path(args.out)
    model, device, ckpt = load_model(out, cpu=args.cpu)
    ds = RivetShapeDataset(out, args.split, heat_sigma=float(ckpt.get("heat_sigma", 4.0)))
    rows_out: List[Dict[str, object]] = []
    total_tp = total_fp = total_fn = 0
    all_errors: List[float] = []
    ious_by_class: Dict[str, List[float]] = {c: [] for c in SHAPE_CLASSES}
    per_class_totals = {c: {"tp": 0, "fp": 0, "fn": 0, "errors": []} for c in SHAPE_CLASSES}

    t0 = time.perf_counter()
    frame_count = 0

    for i in tqdm(range(len(ds)), desc=f"Evaluating {args.split}"):
        x, gt_mask_t, gt_heat_t, sample_id = ds[i]
        img = x.unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_inf0 = time.perf_counter()
        with torch.no_grad():
            logits = model(img)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inf_ms = (time.perf_counter() - t_inf0) * 1000.0
        seg_logits, heat_logits = split_logits(logits)
        seg_probs = torch.softmax(seg_logits, dim=1)[0].detach().cpu().numpy()
        heat_probs = torch.sigmoid(heat_logits)[0].detach().cpu().numpy()
        pred_mask = np.argmax(seg_probs, axis=0).astype(np.uint8)
        gt_mask = gt_mask_t.numpy().astype(np.uint8)

        label_path = out / ds.rows[i]["label"]
        with label_path.open("r", encoding="utf-8") as f:
            lab = json.load(f)
        gt_centers = lab.get("centers", [])

        pred_centers = extract_centers_multiclass(
            seg_probs,
            heat_probs,
            heat_thresh=float(args.heat_thresh),
            mask_thresh=float(args.mask_thresh),
            min_dist=int(args.min_dist),
        )
        m = match_centers_multiclass(pred_centers, gt_centers, float(args.match_dist))

        total_tp += int(m["tp"])
        total_fp += int(m["fp"])
        total_fn += int(m["fn"])
        all_errors.extend(m["errors"])
        for c in SHAPE_CLASSES:
            pc = m["per_class"][c]
            per_class_totals[c]["tp"] += int(pc["tp"])
            per_class_totals[c]["fp"] += int(pc["fp"])
            per_class_totals[c]["fn"] += int(pc["fn"])
            per_class_totals[c]["errors"].extend(pc["errors"])
            iou = class_iou(pred_mask, gt_mask, CLASS_TO_ID[c])
            if not math.isnan(iou):
                ious_by_class[c].append(iou)

        frame_count += 1
        rows_out.append({
            "sample_id": sample_id,
            "source_class": ds.rows[i]["source_class"],
            "gt_count": len(gt_centers),
            "pred_count": len(pred_centers),
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
            "precision": f"{m['precision']:.5f}",
            "recall": f"{m['recall']:.5f}",
            "f1": f"{m['f1']:.5f}",
            "mean_center_error_px": "" if math.isnan(m["mean_error"]) else f"{m['mean_error']:.4f}",
            "median_center_error_px": "" if math.isnan(m["median_error"]) else f"{m['median_error']:.4f}",
            "iou_circle": "" if not ious_by_class["circle"] else f"{ious_by_class['circle'][-1]:.5f}" if int(ds.rows[i]["source_class"] == "circle") else "",
            "iou_oval": "" if not ious_by_class["oval"] else f"{ious_by_class['oval'][-1]:.5f}" if int(ds.rows[i]["source_class"] == "oval") else "",
            "iou_damaged": "" if not ious_by_class["damaged"] else f"{ious_by_class['damaged'][-1]:.5f}" if int(ds.rows[i]["source_class"] == "damaged") else "",
            "inference_ms": f"{inf_ms:.3f}",
        })

    elapsed = time.perf_counter() - t0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_err = float(np.mean(all_errors)) if all_errors else float("nan")
    med_err = float(np.median(all_errors)) if all_errors else float("nan")
    p95_err = float(np.percentile(all_errors, 95)) if all_errors else float("nan")
    mean_iou_by_class = {c: (float(np.mean(vals)) if vals else float("nan")) for c, vals in ious_by_class.items()}
    mean_iou_vals = [v for v in mean_iou_by_class.values() if not math.isnan(v)]
    mean_iou = float(np.mean(mean_iou_vals)) if mean_iou_vals else float("nan")
    fps = frame_count / elapsed if elapsed > 0 else 0.0

    report_csv = out / "reports" / f"eval_shapes_{args.split}.csv"
    with report_csv.open("w", newline="", encoding="utf-8") as f:
        if rows_out:
            writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)

    summary_csv = out / "reports" / f"eval_shapes_{args.split}_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["class", "tp", "fp", "fn", "precision", "recall", "f1", "mean_center_error_px", "mean_iou"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in SHAPE_CLASSES:
            pc = per_class_totals[c]
            tp, fp, fn = pc["tp"], pc["fp"], pc["fn"]
            pr = tp / (tp + fp) if (tp + fp) else 0.0
            rc = tp / (tp + fn) if (tp + fn) else 0.0
            f1c = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
            errs = pc["errors"]
            writer.writerow({
                "class": c,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": f"{pr:.5f}",
                "recall": f"{rc:.5f}",
                "f1": f"{f1c:.5f}",
                "mean_center_error_px": "" if not errs else f"{float(np.mean(errs)):.4f}",
                "mean_iou": "" if math.isnan(mean_iou_by_class[c]) else f"{mean_iou_by_class[c]:.5f}",
            })
        writer.writerow({
            "class": "overall",
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": f"{precision:.5f}",
            "recall": f"{recall:.5f}",
            "f1": f"{f1:.5f}",
            "mean_center_error_px": "" if math.isnan(mean_err) else f"{mean_err:.4f}",
            "mean_iou": "" if math.isnan(mean_iou) else f"{mean_iou:.5f}",
        })

    print("\n=== Multi-class evaluation summary ===")
    print(f"Split:             {args.split}")
    print(f"TP / FP / FN:      {total_tp} / {total_fp} / {total_fn}")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"F1:                {f1:.4f}")
    print(f"Mean center error: {mean_err:.3f} px at {IMG_W}x{IMG_H}")
    print(f"Median error:      {med_err:.3f} px")
    print(f"95% error:         {p95_err:.3f} px")
    print(f"Mean IoU:          {mean_iou:.4f}")
    print(f"Offline FPS:       {fps:.2f}")
    print("\nPer-class summary:")
    for c in SHAPE_CLASSES:
        pc = per_class_totals[c]
        tp, fp, fn = pc["tp"], pc["fp"], pc["fn"]
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1c = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
        err = float(np.mean(pc["errors"])) if pc["errors"] else float("nan")
        iou = mean_iou_by_class[c]
        print(f"  {c:8s}: P={pr:.3f} R={rc:.3f} F1={f1c:.3f} err={err:.2f}px IoU={iou:.3f}")
    print(f"\nReport:            {report_csv}")
    print(f"Summary:           {summary_csv}")


# -----------------------------
# Live camera / ESP32 stream
# -----------------------------

def parse_source(src: str):
    if src.isdigit():
        return int(src)
    return src


@torch.no_grad()
def live_camera(args: argparse.Namespace) -> None:
    out = Path(args.out)
    model, device, ckpt = load_model(out, cpu=args.cpu)
    source = parse_source(str(args.source))
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera/stream: {args.source}")

    print("Press q to quit.")
    last = time.perf_counter()
    fps_smooth = 0.0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Frame read failed.")
            break
        h0, w0 = frame.shape[:2]
        seg_probs, heat_probs = predict_one(model, device, frame)
        centers = extract_centers_multiclass(
            seg_probs,
            heat_probs,
            heat_thresh=float(args.heat_thresh),
            mask_thresh=float(args.mask_thresh),
            min_dist=int(args.min_dist),
        )
        sx, sy = w0 / IMG_W, h0 / IMG_H
        overlay = frame.copy()
        counts = {c: 0 for c in SHAPE_CLASSES}
        for c in centers:
            class_name = str(c["class_name"])
            counts[class_name] += 1
            color = CLASS_COLORS.get(class_name, (0, 255, 255))
            ox, oy = int(round(float(c["x"]) * sx)), int(round(float(c["y"]) * sy))
            cv2.circle(overlay, (ox, oy), 7, color, 2)
            cv2.circle(overlay, (ox, oy), 2, (0, 0, 255), -1)
            cv2.putText(overlay, class_name[0].upper(), (ox + 5, oy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        now = time.perf_counter()
        inst_fps = 1.0 / max(1e-6, now - last)
        last = now
        fps_smooth = inst_fps if fps_smooth == 0.0 else 0.9 * fps_smooth + 0.1 * inst_fps
        msg = f"C={counts['circle']} O={counts['oval']} D={counts['damaged']} total={len(centers)} FPS={fps_smooth:.1f}"
        cv2.putText(overlay, msg, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        cv2.imshow("Rivet live detection - circle/oval/damaged", overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


# -----------------------------
# Hough baseline, optional
# -----------------------------

def hough_detect_centers(img_bgr: np.ndarray, min_radius: int = 6, max_radius: int = 22,
                         dp: float = 1.2, min_dist: int = 14,
                         param1: float = 100, param2: float = 18) -> List[Tuple[float, float, float]]:
    img_r = cv2.resize(img_bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
                               param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    if circles is None:
        return []
    circles = np.round(circles[0, :]).astype(np.float32)
    return [(float(x), float(y), 1.0) for x, y, r in circles]


def hough_baseline(args: argparse.Namespace) -> None:
    out = Path(args.out)
    class_dirs = load_sources(out)
    records = collect_image_records(class_dirs)
    rec_by_key = {str(r["image_key"]): r for r in records}
    ds = RivetShapeDataset(out, args.split)
    total_tp = total_fp = total_fn = 0
    errors: List[float] = []

    print("Note: Hough baseline detects circular centers only. For offline class-aware metrics, predictions are assigned the source folder class.")
    for row in tqdm(ds.rows, desc=f"Hough {args.split}"):
        src_key = row["source_image_key"]
        source_class = row["source_class"]
        if src_key not in rec_by_key:
            continue
        img = cv2.imread(str(rec_by_key[src_key]["path"]))
        if img is None:
            continue
        raw_pred = hough_detect_centers(
            img,
            min_radius=int(args.min_radius), max_radius=int(args.max_radius), dp=float(args.dp),
            min_dist=int(args.min_dist), param1=float(args.param1), param2=float(args.param2),
        )
        pred = [{"x": x, "y": y, "conf": conf, "class_name": source_class, "class_id": CLASS_TO_ID[source_class]} for x, y, conf in raw_pred]
        label_path = out / row["label"]
        with label_path.open("r", encoding="utf-8") as f:
            lab = json.load(f)
        gt = lab.get("centers", [])
        m = match_centers_multiclass(pred, gt, float(args.match_dist))
        total_tp += int(m["tp"])
        total_fp += int(m["fp"])
        total_fn += int(m["fn"])
        errors.extend(m["errors"])

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print("\n=== Hough baseline ===")
    print(f"TP / FP / FN:      {total_tp} / {total_fp} / {total_fn}")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"F1:                {f1:.4f}")
    print(f"Mean center error: {np.mean(errors) if errors else float('nan'):.3f} px at {IMG_W}x{IMG_H}")


# -----------------------------
# Command-line interface
# -----------------------------

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-class rivet center labeling, augmentation, U-Net training, evaluation, and live webcam/ESP32 inference")
    sub = p.add_subparsers(dest="cmd", required=True)

    ap = sub.add_parser("autolabel", help="Automatically create labels.csv using classical detection, then save overlay images for review")
    ap.add_argument("--out", required=True)
    ap.add_argument("--class_dir", action="append", default=None,
                    help="Repeat this as class=folder. Example: --class_dir circle=\"C:\\Users\\Will\\Downloads\\Photos-3-circle\"")
    ap.add_argument("--max_width", type=int, default=1280)
    ap.add_argument("--min_radius", type=int, default=18, help="Minimum rivet radius after resizing to --max_width")
    ap.add_argument("--max_radius", type=int, default=50, help="Maximum rivet radius after resizing to --max_width")
    ap.add_argument("--hough_votes", type=int, default=20, help="Lower = more detections, higher = stricter")
    ap.add_argument("--multi_pass", action="store_true", help="Try several Hough thresholds")
    ap.add_argument("--score_threshold", type=float, default=0.055)
    ap.add_argument("--merge_factor", type=float, default=0.95)
    ap.add_argument("--merge_min_distance", type=float, default=30.0)
    ap.add_argument("--row_tolerance", type=float, default=None)
    ap.add_argument("--no_row_filter", action="store_true")
    ap.add_argument("--front_start", type=float, default=0.18)
    ap.add_argument("--front_end", type=float, default=0.88)
    ap.add_argument("--use_ellipse_fallback", action="store_true")
    ap.add_argument("--min_count", type=int, default=1, help="Images below this detected count are flagged suspect")
    ap.add_argument("--max_count", type=int, default=999, help="Images above this detected count are flagged suspect")
    ap.add_argument("--suspect_score", type=float, default=0.070, help="Images below this mean score are flagged suspect")
    ap.add_argument("--overwrite_existing", action="store_true", help="Overwrite existing labels.csv entries. Default skips already labeled images.")
    ap.add_argument("--debug", action="store_true", help="Save ROI/edge debug images")
    ap.set_defaults(func=auto_label_images)

    cp = sub.add_parser("make_bad_csv", help="Create a CSV checklist from auto_label_report.csv; mark fix=1 for images to manually correct")
    cp.add_argument("--out", required=True)
    cp.add_argument("--report", default=None, help="Optional path to auto_label_report.csv. Default: OUT/auto_label_report.csv")
    cp.add_argument("--csv", default=None, help="Output checklist CSV. Default: OUT/bad_review_list.csv")
    cp.add_argument("--suspects_only", action="store_true", help="Only include rows already flagged as suspect")
    cp.add_argument("--prefill_suspects", action="store_true", help="Put fix=1 automatically for suspect rows")
    cp.set_defaults(func=write_bad_review_template)

    lp = sub.add_parser("label", help="Manually click centers; class is automatically taken from each folder")
    lp.add_argument("--out", required=True)
    lp.add_argument("--class_dir", action="append", default=None,
                    help="Repeat this as class=folder. Example: --class_dir circle=\"C:\\Users\\Will\\Downloads\\Photos-3-circle\"")
    lp.add_argument("--radius", type=float, default=DEFAULT_ORIGINAL_RADIUS)
    lp.add_argument("--start_index", type=int, default=0)
    lp.add_argument("--max_display_w", type=int, default=1500)
    lp.add_argument("--max_display_h", type=int, default=950)
    lp.add_argument("--only", action="append", default=None,
                    help="Open only images whose image_key or filename contains this text. Can repeat.")
    lp.add_argument("--suspects_only", action="store_true",
                    help="Open only images flagged as suspect in auto_label_report.csv")
    lp.add_argument("--bad_csv", default=None,
                    help="CSV checklist of bad images. Use make_bad_csv, then put 1/yes/bad/x in the fix column.")
    lp.set_defaults(func=label_images)

    bp = sub.add_parser("build", help="Create train/val/test dataset and augment train to target count")
    bp.add_argument("--out", required=True)
    bp.add_argument("--target_train", type=int, default=1000)
    bp.add_argument("--val_frac", type=float, default=0.15)
    bp.add_argument("--test_frac", type=float, default=0.15)
    bp.add_argument("--balance_classes", action=argparse.BooleanOptionalAction, default=True,
                    help="Default: balance augmented train samples across circle/oval/damaged")
    bp.add_argument("--overwrite", action="store_true")
    bp.set_defaults(func=build_dataset)

    tp = sub.add_parser("train", help="Train U-Net with class segmentation + class-specific center heatmaps")
    tp.add_argument("--out", required=True)
    tp.add_argument("--epochs", type=int, default=80)
    tp.add_argument("--batch", type=int, default=4)
    tp.add_argument("--lr", type=float, default=1e-3)
    tp.add_argument("--base", type=int, default=32)
    tp.add_argument("--heat_sigma", type=float, default=4.0)
    tp.add_argument("--bg_weight", type=float, default=0.20)
    tp.add_argument("--workers", type=int, default=0)  # Use 0 on Windows unless you know multiprocessing is configured.
    tp.add_argument("--patience", type=int, default=15)
    tp.add_argument("--cpu", action="store_true")
    tp.set_defaults(func=train_model)

    ep = sub.add_parser("eval", help="Evaluate center error, precision/recall/F1, per-class IoU, and offline FPS")
    ep.add_argument("--out", required=True)
    ep.add_argument("--split", default="test", choices=["train", "val", "test"])
    ep.add_argument("--heat_thresh", type=float, default=0.30)
    ep.add_argument("--mask_thresh", type=float, default=0.40)
    ep.add_argument("--min_dist", type=int, default=8)
    ep.add_argument("--match_dist", type=float, default=12.0)
    ep.add_argument("--cpu", action="store_true")
    ep.set_defaults(func=evaluate_model)

    hp = sub.add_parser("hough", help="Optional Hough baseline on val/test split")
    hp.add_argument("--out", required=True)
    hp.add_argument("--split", default="test", choices=["val", "test"])
    hp.add_argument("--min_radius", type=int, default=6)
    hp.add_argument("--max_radius", type=int, default=22)
    hp.add_argument("--dp", type=float, default=1.2)
    hp.add_argument("--min_dist", type=int, default=14)
    hp.add_argument("--param1", type=float, default=100)
    hp.add_argument("--param2", type=float, default=18)
    hp.add_argument("--match_dist", type=float, default=12.0)
    hp.set_defaults(func=hough_baseline)

    vp = sub.add_parser("live", help="Run live camera / ESP32 stream inference")
    vp.add_argument("--out", required=True)
    vp.add_argument("--source", default="0", help="0 for webcam, or ESP32 URL such as http://IP:81/stream")
    vp.add_argument("--heat_thresh", type=float, default=0.30)
    vp.add_argument("--mask_thresh", type=float, default=0.40)
    vp.add_argument("--min_dist", type=int, default=8)
    vp.add_argument("--cpu", action="store_true")
    vp.set_defaults(func=live_camera)

    return p


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
