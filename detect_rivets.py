"""
GPU-accelerated low-contrast rivet detector (PyTorch + OpenCV) for INDE 599.

Folder mode:
    python detect_rivets.py --image-dir "C:/Users/Will/Desktop/INDE 599/Photos-3-001" --output-dir outputs --debug
    
This script keeps the classical CV baseline logic, but moves core image processing
steps to CUDA via PyTorch when available (e.g., RTX 3080).
"""


from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Detection:
    cx: float
    cy: float
    radius: float
    source: str


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def resize_keep_aspect(image: np.ndarray, max_width: int = 1280) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= max_width:
        return image.copy()
    scale = max_width / float(w)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def bgr_to_torch_gray(image_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    img = torch.from_numpy(image_bgr).to(device=device, dtype=torch.float32) / 255.0
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    gray = 0.114 * b + 0.587 * g + 0.299 * r
    return gray.unsqueeze(0).unsqueeze(0)


def torch_gray_to_u8(gray_1x1hw: torch.Tensor) -> np.ndarray:
    out = gray_1x1hw.squeeze().detach().clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy()
    return out


def blue_mask_hsv(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([85, 35, 25], dtype=np.uint8)
    upper = np.array([145, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def auto_face_roi(mask_blue: np.ndarray, full_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    h, w = full_shape
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, int(0.5 * h), w, int(0.45 * h)

    c = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(c)

    roi_x = max(0, x - int(0.03 * w))
    roi_w = min(w - roi_x, bw + int(0.06 * w))
    roi_y = min(h - 1, y + int(0.35 * bh))
    roi_h = min(h - roi_y, int(0.62 * bh))

    if roi_h < 40 or roi_w < 80:
        roi_y = int(0.55 * h)
        roi_h = int(0.38 * h)
        roi_x = 0
        roi_w = w
    return roi_x, roi_y, roi_w, roi_h


def gaussian_kernel(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    k2d = torch.outer(g, g)
    return k2d.unsqueeze(0).unsqueeze(0)


def blur_gpu(gray: torch.Tensor, size: int, sigma: float) -> torch.Tensor:
    kernel = gaussian_kernel(size, sigma, gray.device)
    pad = size // 2
    return F.conv2d(F.pad(gray, (pad, pad, pad, pad), mode="reflect"), kernel)


def morph_open_gpu(gray: torch.Tensor, k: int) -> torch.Tensor:
    pad = k // 2
    eroded = -F.max_pool2d(-gray, kernel_size=k, stride=1, padding=pad)
    opened = F.max_pool2d(eroded, kernel_size=k, stride=1, padding=pad)
    return opened


def sobel_edges_gpu(gray: torch.Tensor, low: float, high: float) -> np.ndarray:
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=gray.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=gray.device).view(1, 1, 3, 3)
    gx = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), kx)
    gy = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), ky)
    mag = torch.sqrt(gx * gx + gy * gy)
    mag = mag / (mag.max() + 1e-6)
    strong = (mag >= high).to(torch.uint8)
    weak = (mag >= low).to(torch.uint8)
    edges = (strong | weak).squeeze().mul(255).cpu().numpy().astype(np.uint8)
    return edges


def preprocess_roi_gpu(roi_bgr: np.ndarray, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray_cv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    gray_t = bgr_to_torch_gray(roi_bgr, device)
    denoised_t = blur_gpu(gray_t, size=7, sigma=1.8)
    bg_t = morph_open_gpu(denoised_t, k=31)
    corrected_t = (denoised_t - bg_t).clamp(0.0, 1.0)

    enhanced_t = (corrected_t / (corrected_t.max() + 1e-6)).clamp(0.0, 1.0)

    enhanced_u8 = torch_gray_to_u8(enhanced_t)
    corrected_u8 = torch_gray_to_u8(corrected_t)
    return gray_cv, enhanced_u8, corrected_u8


def detect_hough(corrected: np.ndarray, min_radius: int, max_radius: int) -> List[Detection]:
    circles = cv2.HoughCircles(
        corrected,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(12, int(1.5 * min_radius)),
        param1=120,
        param2=14,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    out: List[Detection] = []
    if circles is not None:
        for c in np.round(circles[0, :]).astype(int):
            out.append(Detection(float(c[0]), float(c[1]), float(c[2]), "hough"))
    return out


def detect_ellipses(edges: np.ndarray, min_radius: int, max_radius: int) -> List[Detection]:
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    dets: List[Detection] = []
    for cnt in contours:
        if len(cnt) < 20:
            continue
        area = cv2.contourArea(cnt)
        if area < math.pi * (min_radius**2) * 0.25:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim <= 0:
            continue
        circularity = (4.0 * math.pi * area) / (perim * perim)
        if circularity < 0.45:
            continue
        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (ma, mi), _ = ellipse
        major, minor = max(ma, mi), min(ma, mi)
        radius = 0.25 * (major + minor)
        if not (min_radius <= radius <= max_radius * 1.3):
            continue
        if major / (minor + 1e-6) > 2.0:
            continue
        dets.append(Detection(cx, cy, radius, "ellipse"))
    return dets


def merge_detections(detections: Sequence[Detection], overlap_factor: float = 0.65) -> List[Detection]:
    kept: List[Detection] = []
    detections_sorted = sorted(detections, key=lambda d: (d.source != "hough", -d.radius))
    for d in detections_sorted:
        if all(math.hypot(d.cx - k.cx, d.cy - k.cy) >= overlap_factor * min(d.radius, k.radius) for k in kept):
            kept.append(d)
    kept.sort(key=lambda d: d.cx)
    return kept


def annotate(image_bgr: np.ndarray, detections: Sequence[Detection], roi_xywh: Tuple[int, int, int, int], draw_labels: bool) -> np.ndarray:
    out = image_bgr.copy()
    x0, y0, rw, rh = roi_xywh
    cv2.rectangle(out, (x0, y0), (x0 + rw, y0 + rh), (0, 255, 255), 2)
    for i, d in enumerate(detections, start=1):
        cx, cy, r = int(round(x0 + d.cx)), int(round(y0 + d.cy)), int(round(d.radius))
        cv2.circle(out, (cx, cy), max(2, r), (0, 255, 0), 2)
        cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)
        if draw_labels:
            cv2.putText(out, f"{i}: ({cx},{cy})", (cx + 8, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPU rivet detection for low-contrast blue rivets.")
    p.add_argument("--image", help="Single image path.")
    p.add_argument("--image-dir", help="Folder containing images for batch processing.")
    p.add_argument("--output", default="outputs/annotated_result.png", help="Annotated output path for single-image mode.")
    p.add_argument("--output-dir", default="outputs", help="Output folder for batch mode and debug images.")
    p.add_argument("--debug", action="store_true", help="Save debug images.")
    p.add_argument("--min-radius", type=int, default=12)
    p.add_argument("--max-radius", type=int, default=45)
    p.add_argument("--select-roi", action="store_true")
    p.add_argument("--max-width", type=int, default=1280)
    p.add_argument("--no-labels", action="store_true")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Processing device.")
    args = p.parse_args()

    if not args.image and not args.image_dir:
        p.error("You must provide --image OR --image-dir.")
    if args.image and args.image_dir:
        p.error("Use either --image OR --image-dir (not both).")
    return args


def list_images(folder: str) -> List[Path]:
    allowed = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted([p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in allowed])
    if not files:
        raise FileNotFoundError(f"No supported images found in folder: {folder}")
    return files


def run_one(image_path: str, output_path: str, args: argparse.Namespace, device: torch.device) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image_resized = resize_keep_aspect(image, args.max_width)
    h, w = image_resized.shape[:2]
    mask = blue_mask_hsv(image_resized)

    if args.select_roi:
        roi = cv2.selectROI("Select front-face ROI", image_resized, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
        x, y, rw, rh = map(int, roi)
        if rw <= 0 or rh <= 0:
            raise RuntimeError("ROI selection cancelled or invalid.")
    else:
        x, y, rw, rh = auto_face_roi(mask, (h, w))

    roi_bgr = image_resized[y : y + rh, x : x + rw]
    _, enhanced, corrected = preprocess_roi_gpu(roi_bgr, device)
    edges = sobel_edges_gpu(torch.from_numpy(corrected).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0, 0.12, 0.22)

    detections = merge_detections(
        detect_hough(corrected, args.min_radius, args.max_radius)
        + detect_ellipses(edges, args.min_radius, args.max_radius)
    )

    annotated = annotate(image_resized, detections, (x, y, rw, rh), draw_labels=not args.no_labels)
    ensure_dir(os.path.dirname(output_path) or ".")
    cv2.imwrite(output_path, annotated)

    if args.debug:
        stem = Path(output_path).stem
        ddir = Path(args.output_dir)
        ensure_dir(str(ddir))
        cv2.imwrite(str(ddir / f"{stem}_debug_blue_mask.png"), mask)
        cv2.imwrite(str(ddir / f"{stem}_debug_roi.png"), roi_bgr)
        cv2.imwrite(str(ddir / f"{stem}_debug_enhanced.png"), enhanced)
        cv2.imwrite(str(ddir / f"{stem}_debug_edges.png"), edges)

    print(f"\nImage: {image_path}")
    print("Detected rivets:")
    for i, d in enumerate(detections, start=1):
        gx, gy = int(round(x + d.cx)), int(round(y + d.cy))
        print(f"{i}: center=({gx}, {gy}), radius={d.radius:.1f}, source={d.source}")
    print(f"Saved annotated result to: {output_path}")


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.image_dir:
        ensure_dir(args.output_dir)
        for img in list_images(args.image_dir):
            out = str(Path(args.output_dir) / f"{img.stem}_annotated.png")
            run_one(str(img), out, args, device)
    else:
        run_one(args.image, args.output, args, device)


if __name__ == "__main__":
    main()