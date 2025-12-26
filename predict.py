from __future__ import annotations

import argparse
import base64
import io
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO


ImageInput = Union[str, bytes, Image.Image, np.ndarray]


@dataclass
class Detection:
    bbox_raw: List[float]
    bbox_padded: List[float]
    conf: float


def load_image(image: ImageInput, grayscale: bool) -> Image.Image:
    if image is None:
        raise TypeError("Image is required")
    if isinstance(image, str):
        img = Image.open(image)
    elif isinstance(image, bytes):
        img = Image.open(io.BytesIO(image))
    elif isinstance(image, Image.Image):
        img = image
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    elif isinstance(image, dict):
        if "path" in image and image["path"]:
            img = Image.open(image["path"])
        elif "name" in image and image["name"] and os.path.exists(image["name"]):
            img = Image.open(image["name"])
        elif "data" in image and image["data"]:
            data = image["data"]
            if isinstance(data, str):
                payload = data.split(",", 1)[1] if "," in data else data
                img = Image.open(io.BytesIO(base64.b64decode(payload)))
            elif isinstance(data, (bytes, bytearray)):
                img = Image.open(io.BytesIO(data))
            else:
                raise TypeError("Unsupported image input type")
        else:
            raise TypeError("Unsupported image input type")
    elif hasattr(image, "image"):
        img = image.image
    elif hasattr(image, "path"):
        img = Image.open(image.path)
    else:
        raise TypeError("Unsupported image input type")

    if img.mode != "RGB":
        img = img.convert("RGB")
    if grayscale:
        img = img.convert("L").convert("RGB")
    return img


def pad_bbox(bbox: Tuple[float, float, float, float], w: int, h: int, px: float, py: float) -> List[float]:
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * px
    pad_y = bh * py
    x1p = max(0.0, x1 - pad_x)
    y1p = max(0.0, y1 - pad_y)
    x2p = min(float(w), x2 + pad_x)
    y2p = min(float(h), y2 + pad_y)
    return [float(x1p), float(y1p), float(x2p), float(y2p)]


def clean_background(
    img: Image.Image,
    area_percentile: float = 50.0,
    color_clean: bool = False,
    hsv_clean: bool = False,
    hsv_h_min: int = 113,
    hsv_h_max: int = 156,
    hsv_s_min: int = 0,
    hsv_v_min: int = 7,
) -> Image.Image:
    rgb = np.array(img)
    area_percentile = max(0.0, min(100.0, area_percentile))

    mask = None
    if hsv_clean:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h_min = max(0, min(179, int(hsv_h_min)))
        h_max = max(0, min(179, int(hsv_h_max)))
        s_min = max(0, min(255, int(hsv_s_min)))
        v_min = max(0, min(255, int(hsv_v_min)))
        if h_min <= h_max:
            lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
            upper = np.array([h_max, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper).astype(np.uint8)
        else:
            lower1 = np.array([0, s_min, v_min], dtype=np.uint8)
            upper1 = np.array([h_max, 255, 255], dtype=np.uint8)
            lower2 = np.array([h_min, s_min, v_min], dtype=np.uint8)
            upper2 = np.array([179, 255, 255], dtype=np.uint8)
            mask1 = cv2.inRange(hsv, lower1, upper1).astype(np.uint8)
            mask2 = cv2.inRange(hsv, lower2, upper2).astype(np.uint8)
            mask = cv2.bitwise_or(mask1, mask2)

        h, w = rgb.shape[:2]
        min_pixels = max(10, int(0.001 * h * w))
        if np.count_nonzero(mask) < min_pixels:
            lower = np.array([0, s_min, v_min], dtype=np.uint8)
            upper = np.array([179, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper).astype(np.uint8)
    elif color_clean:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        data = lab.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1.0)
        _, labels_k, centers = cv2.kmeans(data, 3, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        centers = centers.astype(np.float32)

        chroma = np.sqrt((centers[:, 1] - 128.0) ** 2 + (centers[:, 2] - 128.0) ** 2)
        if chroma.max() > 8.0:
            target = int(np.argmax(chroma))
            mask = (labels_k.reshape(lab.shape[:2]) == target).astype(np.uint8)

    if mask is None:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, bin_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        foreground = (bin_img == 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
        if num_labels <= 2:
            return Image.fromarray(rgb)

        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
        thresh = max(12.0, float(np.percentile(areas, area_percentile)))
        keep_labels = {i + 1 for i, area in enumerate(areas) if area > thresh}
        mask = np.isin(labels, list(keep_labels))
    else:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if num_labels <= 1:
            return Image.fromarray(rgb)

        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
        thresh = max(12.0, float(np.percentile(areas, area_percentile)))
        keep_labels = {i + 1 for i, area in enumerate(areas) if area > thresh}
        mask = np.isin(labels, list(keep_labels))

    out = np.full_like(rgb, 255, dtype=np.uint8)
    out[mask] = rgb[mask]
    return Image.fromarray(out)


def predict(
    model: YOLO,
    image: ImageInput,
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 300,
    padding_x: float = 0.10,
    padding_y: float = 0.15,
    grayscale_inference: bool = False,
) -> Dict[str, Any]:
    img = load_image(image, grayscale_inference)
    w, h = img.size

    results = model.predict(
        source=img,
        conf=conf,
        iou=iou,
        max_det=max_det,
        verbose=False,
    )

    detections: List[Detection] = []
    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
        confs = results[0].boxes.conf.cpu().numpy().tolist()
        for bbox, score in zip(boxes, confs):
            x1, y1, x2, y2 = bbox
            raw = [float(x1), float(y1), float(x2), float(y2)]
            padded = pad_bbox((x1, y1, x2, y2), w, h, padding_x, padding_y)
            detections.append(Detection(raw, padded, float(score)))

    return {
        "width": w,
        "height": h,
        "detections": [d.__dict__ for d in detections],
    }


def crop_and_clean(
    image: Image.Image,
    det: Detection,
    area_percentile: float = 50.0,
    color_clean: bool = False,
    hsv_clean: bool = False,
    hsv_h_min: int = 80,
    hsv_h_max: int = 160,
    hsv_s_min: int = 30,
    hsv_v_min: int = 30,
) -> Image.Image:
    x1, y1, x2, y2 = map(int, det.bbox_padded)
    roi = image.crop((x1, y1, x2, y2))
    return clean_background(
        roi,
        area_percentile=area_percentile,
        color_clean=color_clean,
        hsv_clean=hsv_clean,
        hsv_h_min=hsv_h_min,
        hsv_h_max=hsv_h_max,
        hsv_s_min=hsv_s_min,
        hsv_v_min=hsv_v_min,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument("--image", required=True, help="Image path")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--padding-x", type=float, default=0.10)
    parser.add_argument("--padding-y", type=float, default=0.15)
    parser.add_argument("--grayscale-inference", action="store_true")
    parser.add_argument("--clean", action="store_true", help="Clean background on crops")
    parser.add_argument("--color-clean", action="store_true", help="Use color-based cleanup")
    parser.add_argument("--hsv-clean", action="store_true", help="Use HSV blue mask cleanup")
    parser.add_argument("--hsv-h-min", type=int, default=113)
    parser.add_argument("--hsv-h-max", type=int, default=156)
    parser.add_argument("--hsv-s-min", type=int, default=0)
    parser.add_argument("--hsv-v-min", type=int, default=7)
    parser.add_argument("--clean-area-percentile", type=float, default=50.0)
    parser.add_argument("--out", default="prediction.json")
    args = parser.parse_args()

    model = YOLO(args.model)
    result = predict(
        model,
        args.image,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        padding_x=args.padding_x,
        padding_y=args.padding_y,
        grayscale_inference=args.grayscale_inference,
    )

    img = load_image(args.image, grayscale=False)
    crops_dir = "crops"
    if args.clean:
        import os

        os.makedirs(crops_dir, exist_ok=True)
        for i, det_dict in enumerate(result["detections"]):
            det = Detection(**det_dict)
            cleaned = crop_and_clean(
                img,
                det,
                area_percentile=args.clean_area_percentile,
                color_clean=args.color_clean,
                hsv_clean=args.hsv_clean,
                hsv_h_min=args.hsv_h_min,
                hsv_h_max=args.hsv_h_max,
                hsv_s_min=args.hsv_s_min,
                hsv_v_min=args.hsv_v_min,
            )
            cleaned.save(f"{crops_dir}/sig_{i}.png")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
