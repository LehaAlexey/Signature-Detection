from __future__ import annotations

import io
from contextlib import asynccontextmanager
from typing import List

import gradio as gr
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes
from PIL import Image, ImageDraw
from ultralytics import YOLO

from predict import Detection, crop_and_clean, load_image, predict

model: YOLO | None = None


def get_model() -> YOLO:
    global model
    if model is None:
        model = YOLO("runs/detect/train3/weights/best.pt")
    return model


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_model()
    yield


app = FastAPI(title="Signature Detector", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    iou: float = Query(0.45, ge=0.0, le=1.0),
    max_det: int = Query(300, ge=1, le=1000),
    padding_x: float = Query(0.10, ge=0.0, le=0.5),
    padding_y: float = Query(0.15, ge=0.0, le=0.5),
    grayscale_inference: bool = Query(False),
    clean: bool = Query(False),
    clean_area_percentile: float = Query(50.0, ge=0.0, le=100.0),
    color_clean: bool = Query(False),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    data = await file.read()
    result = predict(
        model,
        data,
        conf=conf,
        iou=iou,
        max_det=max_det,
        padding_x=padding_x,
        padding_y=padding_y,
        grayscale_inference=grayscale_inference,
    )

    if clean:
        img = load_image(data, grayscale=False)
        cleaned: List[bytes] = []
        for det in result["detections"]:
            det_obj = Detection(**det)
            out = crop_and_clean(
                img,
                det_obj,
                area_percentile=clean_area_percentile,
                color_clean=color_clean,
            )
            buf = io.BytesIO()
            out.save(buf, format="PNG")
            cleaned.append(buf.getvalue())
        result["cleaned_count"] = len(cleaned)

    return JSONResponse(result)


@app.post("/predict/pdf")
async def predict_pdf(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    iou: float = Query(0.45, ge=0.0, le=1.0),
    max_det: int = Query(300, ge=1, le=1000),
    padding_x: float = Query(0.10, ge=0.0, le=0.5),
    padding_y: float = Query(0.15, ge=0.0, le=0.5),
    grayscale_inference: bool = Query(False),
    clean: bool = Query(False),
    dpi: int = Query(250, ge=100, le=400),
    clean_area_percentile: float = Query(50.0, ge=0.0, le=100.0),
    color_clean: bool = Query(False),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    data = await file.read()
    pages = convert_from_bytes(data, dpi=dpi)
    results = []
    for img in pages:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result = predict(
            model,
            buf.getvalue(),
            conf=conf,
            iou=iou,
            max_det=max_det,
            padding_x=padding_x,
            padding_y=padding_y,
            grayscale_inference=grayscale_inference,
        )
        if clean:
            cleaned = [
                crop_and_clean(
                    img,
                    Detection(**det),
                    area_percentile=clean_area_percentile,
                    color_clean=color_clean,
                )
                for det in result["detections"]
            ]
            result["cleaned_count"] = len(cleaned)
        results.append(result)
    return JSONResponse(results)


def gradio_ui() -> gr.Blocks:
    def run_ui(
        image,
        conf,
        iou,
        max_det,
        padding_x,
        padding_y,
        grayscale_inference,
        clean,
        clean_area_percentile,
        color_clean,
    ):
        m = get_model()
        result = predict(
            m,
            image,
            conf=conf,
            iou=iou,
            max_det=max_det,
            padding_x=padding_x,
            padding_y=padding_y,
            grayscale_inference=grayscale_inference,
        )
        img = load_image(image, grayscale=False)
        annotated = img.copy()
        draw = ImageDraw.Draw(annotated)
        for det in result["detections"]:
            x1, y1, x2, y2 = det["bbox_padded"]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        cleaned_gallery = []
        if clean:
            for det in result["detections"]:
                det_obj = Detection(**det)
                cleaned_gallery.append(
                    crop_and_clean(
                        img,
                        det_obj,
                        area_percentile=clean_area_percentile,
                        color_clean=color_clean,
                    )
                )
        return np.array(annotated), cleaned_gallery

    with gr.Blocks(title="Signature Detector") as demo:
        gr.Markdown("Signature Detector")
        with gr.Row():
            inp = gr.Image(type="pil", label="Входное изображение")
            out_img = gr.Image(type="numpy", label="Детекции")

        clean = gr.Checkbox(value=False, label="Очистка фона")
        run_btn = gr.Button("Запуск")

        with gr.Accordion("Расширенные настройки", open=False):
            conf = gr.Slider(0.05, 0.9, value=0.15, step=0.01, label="Порог уверенности (conf)")
            iou = gr.Slider(0.05, 0.9, value=0.45, step=0.01, label="Порог IoU (iou)")
            max_det = gr.Slider(1, 500, value=178, step=1, label="Макс. число детекций (max_det)")
            padding_x = gr.Slider(0.0, 0.5, value=0.06, step=0.01, label="Паддинг по ширине (padding_x)")
            padding_y = gr.Slider(0.0, 0.5, value=0.06, step=0.01, label="Паддинг по высоте (padding_y)")
            grayscale_inference = gr.Checkbox(value=False, label="Градации серого (grayscale_inference)")
            clean_area_percentile = gr.Slider(
                0.0,
                100.0,
                value=50.0,
                step=1.0,
                label="Порог площади, перцентиль",
            )
            color_clean = gr.Checkbox(value=False, label="Цветная очистка")

        out_gallery = gr.Gallery(
            label="Очищенные подписи",
            columns=3,
            height=200,
            object_fit="contain",
        )
        run_btn.click(
            run_ui,
            inputs=[
                inp,
                conf,
                iou,
                max_det,
                padding_x,
                padding_y,
                grayscale_inference,
                clean,
                clean_area_percentile,
                color_clean,
            ],
            outputs=[out_img, out_gallery],
        )
    return demo


if __name__ == "__main__":
    gradio_ui().launch()
