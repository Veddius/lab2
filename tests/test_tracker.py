"""Small deterministic geometry checks independent of the supplied videos."""

import unittest

import cv2
import numpy as np

from tracker import PlanarTracker


class PlanarTrackerTest(unittest.TestCase):
    def test_perspective_and_missing_object(self):
        rng = np.random.default_rng(7)
        template = rng.integers(40, 220, size=(240, 320, 3), dtype=np.uint8)
        template = cv2.GaussianBlur(template, (3, 3), 0)
        cv2.putText(template, "PLANAR OBJECT", (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tracker = PlanarTracker(template)

        expected = np.float32([[95, 55], [430, 30], [465, 285], [75, 300]])
        transform = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [319, 0], [319, 239], [0, 239]]), expected
        )
        scene = cv2.warpPerspective(template, transform, (640, 360))
        found = tracker.detect(scene)
        self.assertIsNotNone(found.corners)
        self.assertLess(np.mean(np.linalg.norm(found.corners - expected, axis=1)), 12)

        blank = np.full((360, 640, 3), 100, dtype=np.uint8)
        self.assertIsNone(tracker.detect(blank).corners)


if __name__ == "__main__":
    unittest.main()
