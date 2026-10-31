"""One explicit image geometry shared by training, inference and evaluation."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


POLICY = 'center_letterbox_v1'


@dataclass(frozen=True)
class LetterboxGeometry:
    """Aspect-preserving resize into a centered square canvas.

    Integer resized dimensions are part of the contract.  Separate x/y scales
    record the sub-pixel rounding exactly, so boxes round-trip without silently
    assuming that the rounded resize has one idealized scale.
    """

    canvas_size: int
    source_width: int
    source_height: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    scale_x: float
    scale_y: float

    @classmethod
    def from_source(cls, width: int, height: int, canvas_size: int):
        if (type(width) is not int or type(height) is not int
                or type(canvas_size) is not int or min(width, height) < 1
                or canvas_size < 32):
            raise ValueError('Positive integer source shape and canvas are required')
        ideal = min(canvas_size / width, canvas_size / height)
        resized_width = min(canvas_size, max(1, round(width * ideal)))
        resized_height = min(canvas_size, max(1, round(height * ideal)))
        return cls(
            canvas_size=canvas_size,
            source_width=width,
            source_height=height,
            resized_width=resized_width,
            resized_height=resized_height,
            pad_left=(canvas_size - resized_width) // 2,
            pad_top=(canvas_size - resized_height) // 2,
            scale_x=resized_width / width,
            scale_y=resized_height / height,
        )

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict) or payload.get('policy') != POLICY:
            raise ValueError('Unknown preprocessing geometry')
        expected = cls.from_source(
            payload.get('source_width'), payload.get('source_height'),
            payload.get('canvas_size'))
        if payload != expected.as_dict():
            raise ValueError('Serialized letterbox geometry is internally inconsistent')
        return expected

    def as_dict(self):
        return {
            'policy': POLICY,
            'canvas_size': self.canvas_size,
            'source_width': self.source_width,
            'source_height': self.source_height,
            'resized_width': self.resized_width,
            'resized_height': self.resized_height,
            'pad_left': self.pad_left,
            'pad_top': self.pad_top,
            'scale_x': self.scale_x,
            'scale_y': self.scale_y,
        }

    @staticmethod
    def _boxes(value):
        boxes = np.asarray(value, dtype=np.float64)
        if (boxes.ndim != 2 or boxes.shape[1] != 4
                or not np.isfinite(boxes).all()):
            raise ValueError('Expected finite Nx4 xyxy boxes')
        return boxes

    def to_canvas_xyxy(self, value):
        boxes = self._boxes(value).copy()
        boxes[:, 0::2] = boxes[:, 0::2] * self.scale_x + self.pad_left
        boxes[:, 1::2] = boxes[:, 1::2] * self.scale_y + self.pad_top
        return boxes

    def to_source_xyxy(self, value):
        boxes = self._boxes(value).copy()
        boxes[:, 0::2] = (boxes[:, 0::2] - self.pad_left) / self.scale_x
        boxes[:, 1::2] = (boxes[:, 1::2] - self.pad_top) / self.scale_y
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, self.source_width)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, self.source_height)
        return boxes

    def labels_for_source_xywh(self, bbox, *, small_max=32., medium_max=96.,
                               far_center_y_max=.6):
        if (not isinstance(bbox, (list, tuple, np.ndarray)) or len(bbox) != 4
                or not all(math.isfinite(float(value)) for value in bbox)):
            raise ValueError('Expected one finite xywh box')
        x, y, width, height = (float(value) for value in bbox)
        if (x < 0 or y < 0 or width <= 0 or height <= 0
                or x + width > self.source_width or y + height > self.source_height
                or not 0 < small_max < medium_max
                or not 0 <= far_center_y_max <= 1):
            raise ValueError('Box or geometric thresholds are invalid')
        canvas_width = width * self.scale_x
        canvas_height = height * self.scale_y
        equivalent_side = math.sqrt(canvas_width * canvas_height)
        center_y = ((y + height / 2) * self.scale_y + self.pad_top) / self.canvas_size
        small = equivalent_side <= small_max
        return {
            'small': small,
            'mid_like': small_max < equivalent_side <= medium_max,
            'near_like': equivalent_side > medium_max,
            'far_like': small and center_y <= far_center_y_max,
        }

    def apply_pil(self, image, fill=(114, 114, 114)):
        from PIL import Image

        if image.size != (self.source_width, self.source_height):
            raise ValueError('Image shape differs from letterbox geometry')
        resized = image.resize(
            (self.resized_width, self.resized_height), Image.Resampling.BILINEAR)
        canvas = Image.new(image.mode, (self.canvas_size, self.canvas_size), fill)
        canvas.paste(resized, (self.pad_left, self.pad_top))
        return canvas
