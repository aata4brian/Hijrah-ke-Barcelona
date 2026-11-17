"""Pinned YOLOv8n COCO pose ONNX adapter; no Ultralytics dependency."""
from pathlib import Path
import cv2
import numpy as np

from .common import digest

MODEL_SHA256 = '04f6d2416266f2aba6c5ba8b26de33ed9eba3279f972ac02e33f9f1366547586'
MODEL_URL = ('https://huggingface.co/Xenova/yolov8n-pose/resolve/'
    'da4224085e2f6ce4c9ff9b670e28765194619db2/onnx/model.onnx')


def decode(output, scale, left, top, width, height, confidence=0.30):
    data = np.asarray(output)
    if data.shape != (1, 56, 8400):
        raise ValueError(f'Unsupported pose output {data.shape}; expected [1,56,8400]')
    rows = data[0].T
    rows = rows[np.isfinite(rows).all(axis=1) & (rows[:, 4] >= confidence)]
    boxes, scores, poses = [], [], []
    for row in rows:
        cx, cy, w, h = row[:4]
        x, y = (cx-w/2-left)/scale, (cy-h/2-top)/scale
        w, h = w/scale, h/scale
        x2, y2 = min(width, x+w), min(height, y+h)
        x, y = max(0, x), max(0, y)
        if x2-x < 6 or y2-y < 18:
            continue
        keypoints = row[5:].reshape(17, 3).copy()
        keypoints[:, 0] = (keypoints[:, 0]-left)/scale
        keypoints[:, 1] = (keypoints[:, 1]-top)/scale
        boxes.append([float(x), float(y), float(x2-x), float(y2-y)])
        scores.append(float(row[4]))
        poses.append(keypoints)
    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence, 0.45) if boxes else []
    return [dict(box=boxes[i], score=scores[i], keypoints=poses[i])
        for i in np.asarray(indices, dtype=int).reshape(-1)]


class PoseDetector:
    def __init__(self, model, threads=4):
        self.model = Path(model)
        if digest(self.model) != MODEL_SHA256:
            raise ValueError('Pose model SHA256 mismatch; use setup_model.py')
        cv2.setNumThreads(threads)
        self.net = cv2.dnn.readNetFromONNX(str(model))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def __call__(self, image):
        h, w = image.shape[:2]
        scale = min(640/w, 640/h)
        nw, nh = round(w*scale), round(h*scale)
        left, top = (640-nw)//2, (640-nh)//2
        resized = cv2.resize(image, (nw, nh))
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        canvas[top:top+nh, left:left+nw] = resized
        self.net.setInput(cv2.dnn.blobFromImage(canvas, 1/255, (640,640), swapRB=True))
        return decode(self.net.forward(), scale, left, top, w, h)
