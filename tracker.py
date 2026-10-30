"""Track a textured planar object whose template fills the first video frame."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Detection:
    corners: np.ndarray | None
    matches: int = 0
    inliers: int = 0


class PlanarTracker:
    def __init__(self, template: np.ndarray, min_inliers: int = 8):
        self.height, self.width = template.shape[:2]
        self.min_inliers = min_inliers
        self.detector = cv2.SIFT_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        self.keypoints, self.descriptors = self.detector.detectAndCompute(gray, None)
        if self.descriptors is None or len(self.keypoints) < min_inliers:
            raise ValueError("В первом кадре недостаточно ключевых точек; выберите текстурный объект")
        self.template_corners = np.float32(
            [[0, 0], [self.width - 1, 0], [self.width - 1, self.height - 1], [0, self.height - 1]]
        ).reshape(-1, 1, 2)

    def detect(self, frame: np.ndarray) -> Detection:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 4:
            return Detection(None)

        pairs = self.matcher.knnMatch(self.descriptors, descriptors, k=2)
        good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < 0.8 * n.distance]
        if len(good) < self.min_inliers:
            return Detection(None, len(good))

        source = np.float32([self.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        target = np.float32([keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(source, target, cv2.RANSAC, 4.0)
        inliers = int(mask.sum()) if mask is not None else 0
        detection = Detection(None, len(good), inliers)
        if matrix is None or inliers < self.min_inliers or inliers / len(good) < 0.3:
            return detection

        corners = cv2.perspectiveTransform(self.template_corners, matrix).reshape(4, 2)
        if self._valid_quad(corners, frame.shape[1], frame.shape[0]):
            detection.corners = corners
        return detection

    @staticmethod
    def _valid_quad(corners: np.ndarray, width: int, height: int) -> bool:
        if not np.isfinite(corners).all():
            return False
        polygon = corners.astype(np.float32)
        if not cv2.isContourConvex(polygon):
            return False
        area = abs(cv2.contourArea(polygon))
        if not (0.001 * width * height < area < 2 * width * height):
            return False
        lengths = np.linalg.norm(polygon - np.roll(polygon, -1, axis=0), axis=1)
        if lengths.min() < 8 or lengths.max() / lengths.min() > 25:
            return False
        x, y = polygon[:, 0], polygon[:, 1]
        return bool(
            (x > -0.5 * width).all() and (x < 1.5 * width).all()
            and (y > -0.5 * height).all() and (y < 1.5 * height).all()
        )


def annotate(frame: np.ndarray, detection: Detection, label: str) -> np.ndarray:
    result = frame.copy()
    if detection.corners is not None:
        quad = np.rint(detection.corners).astype(np.int32)
        cv2.polylines(result, [quad], True, (0, 220, 0), 2, cv2.LINE_AA)
        origin = tuple(np.clip(quad[0] + [0, -8], [4, 20], [frame.shape[1] - 1, frame.shape[0] - 1]))
        cv2.putText(result, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(result, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(result, "Object not found", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(result, "Object not found", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return result


def process_video(input_path: Path, output_path: Path, stats_path: Path, label: str, min_inliers: int, roi: tuple[int, int, int, int] | None = None) -> None:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {input_path}")
    ok, first = capture.read()
    if not ok:
        raise RuntimeError("Не удалось прочитать первый кадр")
    if roi is None:
        x, y, roi_width, roi_height = 0, 0, first.shape[1], first.shape[0]
    else:
        x, y, roi_width, roi_height = roi
        if x < 0 or y < 0 or roi_width <= 0 or roi_height <= 0 or x + roi_width > first.shape[1] or y + roi_height > first.shape[0]:
            raise ValueError("ROI выходит за границы первого кадра")
    tracker = PlanarTracker(first[y:y + roi_height, x:x + roi_width], min_inliers)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Не удалось создать видео: {output_path}")

    count = found = 0
    try:
        with stats_path.open("w", newline="", encoding="utf-8") as stream:
            csv_writer = csv.writer(stream)
            csv_writer.writerow(["frame", "time_s", "found", "matches", "inliers", "x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3"])
            frame = first
            while True:
                if count == 0:
                    initial_corners = tracker.template_corners.reshape(4, 2) + np.float32([x, y])
                    detection = Detection(initial_corners, len(tracker.keypoints), len(tracker.keypoints))
                else:
                    detection = tracker.detect(frame)
                found += detection.corners is not None
                coords = [round(float(value), 2) for value in detection.corners.flatten()] if detection.corners is not None else [""] * 8
                csv_writer.writerow([count, round(count / fps, 3), int(detection.corners is not None), detection.matches, detection.inliers, *coords])
                writer.write(annotate(frame, detection, label))
                count += 1
                ok, frame = capture.read()
                if not ok:
                    break
    finally:
        capture.release()
        writer.release()
    print(f"{input_path.name}: {found}/{count} кадров с рамкой; результат: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="AVI/MP4; первый кадр содержит крупный объект")
    parser.add_argument("--output", type=Path, help="Выходной MP4")
    parser.add_argument("--stats", type=Path, help="Покадровая статистика CSV")
    parser.add_argument("--label", default="Object", help="Подпись рамки латиницей")
    parser.add_argument("--min-inliers", type=int, default=8)
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"), help="Область объекта на первом кадре")
    args = parser.parse_args()
    output = args.output or Path("results") / (args.input.stem + "-tracked.mp4")
    stats = args.stats or Path("results") / (args.input.stem + "-stats.csv")
    process_video(args.input, output, stats, args.label, args.min_inliers, tuple(args.roi) if args.roi else None)


if __name__ == "__main__":
    main()
