"""FP32 image-conditioned row gains and row-dependent residual mixing."""
from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


CONDITIONER_SEED_OFFSET = 261025
INITIAL_RESIDUAL_WEIGHT = .1
LEVEL_COORDINATES = (0., .5, 1.)


def _keep_fp32(fn):
    """Move devices without first rounding the retained FP32 parameters."""
    def apply(tensor):
        transformed = fn(tensor)
        if tensor.is_floating_point():
            return tensor.to(device=transformed.device, dtype=torch.float32)
        return transformed
    return apply


def detached_stats(value):
    with torch.no_grad():
        value = value.detach().float()
        return dict(min=float(value.min()), mean=float(value.mean()),
                    max=float(value.max()), std=float(value.std(unbiased=False)))


def _descriptor_stats(value):
    return {name: detached_stats(value[:, index:index + 1])
            for index, name in enumerate(('energy', 'vertical_variation', 'row', 'level'))}


def _validate_feature(value):
    if (not isinstance(value, torch.Tensor) or value.ndim != 4
            or not value.is_floating_point() or any(size < 1 for size in value.shape)):
        raise ValueError('Expected a nonempty floating BCHW feature')


def _validate_features(features):
    if not isinstance(features, (list, tuple)) or len(features) != 3:
        raise ValueError('Head gain expects exactly P3, P4 and P5')
    for value in features:
        _validate_feature(value)
    first = features[0]
    if any(value.shape[0] != first.shape[0] or value.device != first.device
           or value.dtype != first.dtype for value in features):
        raise ValueError('Pyramid batch, dtype and device must agree')
    if any(features[index].shape[axis] < features[index + 1].shape[axis]
           for index in (0, 1) for axis in (-2, -1)):
        raise ValueError('P3/P4/P5 resolutions must decrease monotonically')


class RowConditioner(nn.Module):
    """One pointwise conditioner with channels 4, 8, 1 for rows and levels.

    The denominator is detached, but both content numerators retain gradients.
    Head and residual diagnostics are recorded separately.
    """

    hidden_channels = 8

    def __init__(self, state_id=0):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(state_id) + CONDITIONER_SEED_OFFSET)
            self.conditioner = nn.Sequential(
                nn.Conv2d(4, 8, 1, device='cpu', dtype=torch.float32), nn.SiLU(),
                nn.Conv2d(8, 1, 1, device='cpu', dtype=torch.float32))
            nn.init.zeros_(self.conditioner[-1].weight)
            nn.init.zeros_(self.conditioner[-1].bias)
        self.diagnostics_enabled = False
        self._last_head_diagnostics = None
        self._last_blend_diagnostics = None

    def _apply(self, fn, recurse=True):
        return super()._apply(_keep_fp32(fn), recurse=recurse)

    @staticmethod
    def descriptor(value, q):
        """Return B×4×H×1 channels ordered energy, variation, row, level."""
        _validate_feature(value)
        if q not in LEVEL_COORDINATES:
            raise ValueError('Original pyramid level coordinate must be 0, 0.5 or 1')
        with torch.autocast(device_type=value.device.type, enabled=False):
            value = value.float()
            scale = value.detach().abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
            energy = value.abs().mean(dim=(1, 3), keepdim=True) / scale
            if value.shape[-2] == 1:
                variation = torch.zeros_like(energy)
            else:
                difference = (value[:, :, 1:, :] - value[:, :, :-1, :]).abs()
                variation = difference.mean(dim=(1, 3), keepdim=True) / scale
                variation = torch.cat((variation, variation[:, :, -1:, :]), dim=2)
            row = torch.linspace(-1., 1., value.shape[-2], device=value.device,
                                 dtype=torch.float32).reshape(1, 1, -1, 1)
            row = row.expand(value.shape[0], 1, value.shape[-2], 1)
            level = torch.full_like(row, float(q))
            return torch.cat((energy, variation, row, level), dim=1)

    def _condition(self, value, q):
        with torch.autocast(device_type=value.device.type, enabled=False):
            descriptor = self.descriptor(value, q)
            return self.conditioner(descriptor), descriptor

    def logits(self, value, q):
        return self._condition(value, q)[0]

    def forward(self, features):
        _validate_features(features)
        outputs, gains, descriptors = [], {}, {}
        with torch.autocast(device_type=features[0].device.type, enabled=False):
            for index, (value, q) in enumerate(zip(features, LEVEL_COORDINATES)):
                logits, descriptor = self._condition(value, q)
                gain = 1. + .5 * logits.tanh()
                outputs.append((value.float() * gain).to(value.dtype))
                if self.diagnostics_enabled:
                    level = f'P{index + 3}'
                    gains[level] = detached_stats(gain)
                    descriptors[level] = _descriptor_stats(descriptor)
        if self.diagnostics_enabled:
            self._last_head_diagnostics = dict(gains=gains, descriptors=descriptors)
        return tuple(outputs)

    def mix_residual(self, high, native_low, frequency_high, frequency_low, *, original_low):
        """Compute low logits at stride 16, then resize logits before blending."""
        for value in (high, native_low, frequency_high, frequency_low, original_low):
            _validate_feature(value)
        if high.shape != frequency_high.shape or native_low.shape != frequency_low.shape:
            raise ValueError('Residual frequency branches must match their native branch shapes')
        if any(value.shape[0] != high.shape[0] or value.shape[-2:] != high.shape[-2:]
               or value.device != high.device or value.dtype != high.dtype
               for value in (native_low, frequency_high, frequency_low)):
            raise ValueError('Residual branch batch, resolution, device and dtype must agree')
        if (original_low.shape[:2] != native_low.shape[:2]
                or native_low.shape[-2:] != tuple(2 * size for size in original_low.shape[-2:])
                or original_low.device != high.device or original_low.dtype != high.dtype):
            raise ValueError('Original low feature must match the native 2x upsampling seam')
        outputs, weights, descriptors = [], {}, {}
        with torch.autocast(device_type=high.device.type, enabled=False):
            for name, base, frequency, source, q in (
                    ('high', high, frequency_high, high, 0.),
                    ('low', native_low, frequency_low, original_low, .5)):
                logits, descriptor = self._condition(source, q)
                if name == 'low':
                    logits = F.interpolate(logits, size=(high.shape[-2], 1),
                                           mode='bilinear', align_corners=False)
                weight = (math.log(.1 / .9) + logits).sigmoid()
                outputs.append((base.float() + weight * (frequency.float() - base.float()))
                               .to(base.dtype))
                if self.diagnostics_enabled:
                    weights[name] = detached_stats(weight)
                    descriptors[name] = _descriptor_stats(descriptor)
        if self.diagnostics_enabled:
            self._last_blend_diagnostics = dict(weights=weights, descriptors=descriptors)
        return tuple(outputs)

    def diagnostics(self):
        return copy.deepcopy(dict(enabled=self.diagnostics_enabled,
                                  head=self._last_head_diagnostics,
                                  blend=self._last_blend_diagnostics))
