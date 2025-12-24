from __future__ import annotations

import argparse
import io
import json
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
    if isinstance(image, str):
        img = Image.open(image)
    elif isinstance(image, bytes):
        img = Image.open(io.BytesIO(image))
    elif isinstance(image, Image.Image):
        img = image
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image)
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


def clean_background(img: Image.Image, area_percentile: float = 50.0) -> Image.Image:
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bin_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Foreground is dark strokes.
    foreground = (bin_img == 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    if num_labels <= 2:
        return Image.fromarray(bin_img)

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    area_percentile = max(0.0, min(100.0, area_percentile))
    # Drop more small detached components while keeping thin strokes by area only.
    thresh = max(12.0, float(np.percentile(areas, area_percentile)))
    keep_labels = {i + 1 for i, area in enumerate(areas) if area > thresh}

    mask = np.isin(labels, list(keep_labels))
    out = np.full_like(bin_img, 255, dtype=np.uint8)
    out[mask] = 0
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
    image: Image.Image, det: Detection, area_percentile: float = 50.0
) -> Image.Image:
    x1, y1, x2, y2 = map(int, det.bbox_padded)
    roi = image.crop((x1, y1, x2, y2))
    return clean_background(roi, area_percentile=area_percentile)


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
            cleaned = crop_and_clean(img, det)
            cleaned.save(f"{crops_dir}/sig_{i}.png")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
