"""Per-image positive, background and ignore-region loss weights."""
from dataclasses import dataclass
import math
from numbers import Real
from typing import Mapping

import torch
from torch import Tensor


@dataclass(frozen=True)
class ImageSupervision:
    origin: str
    loss_weight: float
    policy: str

    def __post_init__(self):
        expected = {'human': 'human_positive_and_background_except_ignore',
                    'pseudo': 'positive_only_rest_unknown'}
        if self.origin not in expected or self.policy != expected[self.origin]:
            raise ValueError('Unsupported or missing annotation origin/policy')
        if (isinstance(self.loss_weight, bool) or not isinstance(self.loss_weight, Real)
                or not math.isfinite(self.loss_weight) or not 0 < self.loss_weight <= 1):
            raise ValueError('Loss weight must be finite in (0, 1]')
        if self.origin == 'human' and self.loss_weight != 1:
            raise ValueError('Human loss weight must remain 1')

    @classmethod
    def from_coco_image(cls, image: Mapping) -> 'ImageSupervision':
        """Read the explicit loss-policy fields without inferring label coverage."""
        try:
            return cls(image['maize_annotation_origin'], image['maize_loss_weight'],
                       image['maize_supervision_policy'])
        except KeyError as error:
            raise ValueError(f'Missing supervision provenance: {error.args[0]}') from error


@dataclass(frozen=True)
class SupervisionWeights:
    classification: Tensor
    regression: Tensor


def supervision_weights(supervision: ImageSupervision, predicted_xyxy: Tensor,
                        foreground: Tensor, ignored_xyxy: Tensor,
                        image_size: tuple[int, int]) -> SupervisionWeights:
    """Return detached [N] multipliers in the transformed image canvas."""
    if not isinstance(supervision, ImageSupervision):
        raise ValueError('Validated ImageSupervision required')
    if (len(image_size) != 2 or any(isinstance(x, bool) or not isinstance(x, Real)
            or not math.isfinite(x) or x <= 0 for x in image_size)):
        raise ValueError('Image size must be finite positive (height, width)')
    for name, boxes in [('predicted', predicted_xyxy), ('ignored', ignored_xyxy)]:
        if (not isinstance(boxes, Tensor) or boxes.ndim != 2 or boxes.shape[1] != 4
                or not boxes.is_floating_point() or not torch.isfinite(boxes).all()
                or (boxes[:, 2:] < boxes[:, :2]).any()):
            raise ValueError(f'{name} boxes must be finite ordered floating [N,4] xyxy')
    if (foreground.dtype != torch.bool or foreground.ndim != 1
            or foreground.shape[0] != len(predicted_xyxy)
            or foreground.device != predicted_xyxy.device):
        raise ValueError('Foreground must be Boolean [N] on prediction device')
    positive = foreground.to(dtype=torch.float32)
    if supervision.origin == 'pseudo':
        return SupervisionWeights(positive * supervision.loss_weight,
                                  positive * supervision.loss_weight)
    classification = torch.ones_like(positive)
    if len(ignored_xyxy) and len(predicted_xyxy):
        with torch.no_grad():
            h, w = image_size
            boxes = predicted_xyxy.detach().float().clone()
            boxes[:, 0::2].clamp_(0, w)
            boxes[:, 1::2].clamp_(0, h)
            ignored = ignored_xyxy.detach().to(boxes)
            centres = (boxes[:, :2] + boxes[:, 2:]) * .5
            inside = ((centres[:, None] >= ignored[None, :, :2]) &
                      (centres[:, None] <= ignored[None, :, 2:])).all(-1).any(-1)
            intersection = (torch.minimum(boxes[:, None, 2:], ignored[None, :, 2:]) -
                            torch.maximum(boxes[:, None, :2], ignored[None, :, :2])).clamp(min=0).prod(-1)
            area = (boxes[:, 2:] - boxes[:, :2]).prod(-1).clamp(min=1e-6)
            suppress = (inside | ((intersection / area[:, None]).amax(-1) >= .5)) & ~foreground
            classification[suppress] = 0
    return SupervisionWeights(classification, positive)
