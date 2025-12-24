from __future__ import annotations

import argparse
import ast
import csv
import random
import shutil
from pathlib import Path

from ultralytics import YOLO
from PIL import Image


def write_data_yaml(data_root: Path, out_path: Path) -> None:
    text = "\n".join(
        [
            f"path: {data_root.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: signature",
            "",
        ]
    )
    out_path.write_text(text, encoding="utf-8")


def make_grayscale_dataset(src_root: Path, dst_root: Path) -> None:
    for split in ["train", "val", "test"]:
        src_img_dir = src_root / "images" / split
        if not src_img_dir.exists():
            continue
        dst_img_dir = dst_root / "images" / split
        dst_lbl_dir = dst_root / "labels" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in src_img_dir.rglob("*"):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            img = Image.open(img_path).convert("L")
            img.save(dst_img_dir / img_path.name)

            src_lbl = src_root / "labels" / split / f"{img_path.stem}.txt"
            if src_lbl.exists():
                dst_lbl = dst_lbl_dir / src_lbl.name
                dst_lbl.write_text(src_lbl.read_text(encoding="utf-8"), encoding="utf-8")


def copy_yolo_dataset(src_root: Path, dst_root: Path) -> None:
    for split in ["train", "val", "test"]:
        src_img_dir = src_root / "images" / split
        if not src_img_dir.exists():
            continue
        dst_img_dir = dst_root / "images" / split
        dst_lbl_dir = dst_root / "labels" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in src_img_dir.rglob("*"):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            shutil.copy2(img_path, dst_img_dir / img_path.name)
            src_lbl = src_root / "labels" / split / f"{img_path.stem}.txt"
            dst_lbl = dst_lbl_dir / f"{img_path.stem}.txt"
            if src_lbl.exists():
                dst_lbl.write_text(src_lbl.read_text(encoding="utf-8"), encoding="utf-8")
            else:
                dst_lbl.write_text("", encoding="utf-8")


def load_image_map(image_ids_csv: Path) -> dict[int, tuple[str, float, float]]:
    mapping: dict[int, tuple[str, float, float]] = {}
    with image_ids_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = int(row["id"])
            file_name = row["file_name"]
            width = float(row["width"])
            height = float(row["height"])
            mapping[image_id] = (file_name, width, height)
    return mapping


def parse_bbox(bbox_str: str) -> tuple[float, float, float, float]:
    bbox = ast.literal_eval(bbox_str)
    return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])


def bbox_to_yolo(
    x: float, y: float, w: float, h: float, img_w: float, img_h: float
) -> tuple[float, float, float, float]:
    if max(x, y, w, h) > 1.0:
        x /= img_w
        y /= img_h
        w /= img_w
        h /= img_h
    xc = x + w / 2.0
    yc = y + h / 2.0
    return xc, yc, w, h


def read_annotations(csv_path: Path) -> dict[int, list[tuple[float, float, float, float]]]:
    ann: dict[int, list[tuple[float, float, float, float]]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["category_id"]) != 1:
                continue
            image_id = int(row["image_id"])
            ann.setdefault(image_id, []).append(parse_bbox(row["bbox"]))
    return ann


def build_archive_dataset(
    archive_root: Path,
    out_root: Path,
    seed: int,
    val_ratio: float,
) -> None:
    image_map = load_image_map(archive_root / "image_ids.csv")
    train_ann = read_annotations(archive_root / "train.csv")
    test_ann = read_annotations(archive_root / "test.csv")

    train_ids = list(train_ann.keys())
    rng = random.Random(seed)
    rng.shuffle(train_ids)
    val_count = int(len(train_ids) * val_ratio)
    val_ids = set(train_ids[:val_count])
    train_ids = [i for i in train_ids if i not in val_ids]

    split_ids = {
        "train": train_ids,
        "val": list(val_ids),
        "test": list(test_ann.keys()),
    }
    split_ann = {
        "train": train_ann,
        "val": train_ann,
        "test": test_ann,
    }

    for split, ids in split_ids.items():
        img_dir = out_root / "images" / split
        lbl_dir = out_root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for image_id in ids:
            if image_id not in image_map:
                continue
            file_name, img_w, img_h = image_map[image_id]
            src_img = archive_root / "images" / file_name
            if not src_img.exists():
                continue
            shutil.copy2(src_img, img_dir / file_name)

            labels = []
            for x, y, w, h in split_ann[split].get(image_id, []):
                xc, yc, wn, hn = bbox_to_yolo(x, y, w, h, img_w, img_h)
                labels.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
            (lbl_dir / f"{Path(file_name).stem}.txt").write_text(
                "\n".join(labels), encoding="utf-8"
            )


def build_merged_dataset(
    data_root: Path,
    archive_root: Path,
    out_root: Path,
    seed: int,
    val_ratio: float,
) -> None:
    if out_root.exists():
        shutil.rmtree(out_root)
    copy_yolo_dataset(data_root, out_root)
    build_archive_dataset(archive_root, out_root, seed=seed, val_ratio=val_ratio)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/signature", help="Dataset root dir")
    parser.add_argument("--data", default="", help="Path to data.yaml (optional)")
    parser.add_argument("--grayscale-data", action="store_true", help="Train on grayscale copies")
    parser.add_argument("--merge-archive", action="store_true", help="Merge archive dataset")
    parser.add_argument("--archive-root", default="archive", help="Archive dataset root")
    parser.add_argument("--merged-root", default="data/merged", help="Merged dataset output")
    parser.add_argument("--archive-val-ratio", type=float, default=0.1)
    parser.add_argument("--model", default="yolo11n.pt", help="Base model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="train")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data) if args.data else Path("signature.yaml")
    if not args.data:
        data_root = Path(args.data_root).resolve()
        if args.merge_archive:
            archive_root = Path(args.archive_root).resolve()
            merged_root = Path(args.merged_root).resolve()
            build_merged_dataset(
                data_root,
                archive_root,
                merged_root,
                seed=args.seed,
                val_ratio=args.archive_val_ratio,
            )
            data_root = merged_root
        if args.grayscale_data:
            gray_root = data_root.parent / f"{data_root.name}_gray"
            make_grayscale_dataset(data_root, gray_root)
            write_data_yaml(gray_root, data_path)
        else:
            write_data_yaml(data_root, data_path)

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        seed=args.seed,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=2.0,
        perspective=0.001,
        fliplr=0.5,
        translate=0.05,
        scale=0.4,
    )


if __name__ == "__main__":
    main()
