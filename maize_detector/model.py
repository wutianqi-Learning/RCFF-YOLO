"""YOLO11 with independent row conditioning at the neck and detection head."""
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F
import yaml

os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
import ultralytics
from ultralytics.nn.tasks import DetectionModel
from .conditioning import RowConditioner
from .loss import SupervisedYoloLoss
from .upstream import upstream_module


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class YoloDetector(torch.nn.Module):
    """Single-class YOLO with native matching, box regression and decoding."""
    def __init__(self, family='S11', seed=0, gains=None):
        super().__init__()
        if ultralytics.__version__ != '8.3.160':
            raise ValueError('This audited adapter requires ultralytics==8.3.160')
        variants = {
            'N08': ('v8/yolov8.yaml', 22, 'n'), 'S08': ('v8/yolov8.yaml', 22, 's'),
            'L08': ('v8/yolov8.yaml', 22, 'l'),
            'N11': ('11/yolo11.yaml', 23, 'n'), 'S11': ('11/yolo11.yaml', 23, 's'),
            'L11': ('11/yolo11.yaml', 23, 'l'),
            'N12': ('12/yolo12.yaml', 21, 'n'), 'S12': ('12/yolo12.yaml', 21, 's'),
        }
        if family not in variants:
            raise ValueError(
                'Only audited YOLOv8/11 n/s/l and YOLO12 n/s detectors are implemented')
        suffix, head_index, scale = variants[family]
        self.head_index = head_index
        path = Path(ultralytics.__file__).parent / 'cfg/models' / suffix
        config = yaml.safe_load(path.read_text())
        config.update(scale=scale, nc=1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.native = DetectionModel(config, ch=3, nc=1, verbose=False)
        head = self.native.model[-1]
        if head.nc != 1 or head.reg_max != 16 or head.nl != 3 or getattr(self.native, 'end2end', False):
            raise ValueError('Unexpected native Detect architecture')
        gains = dict(box=7.5, cls=.5, dfl=1.5) if gains is None else dict(gains)
        if set(gains) != {'box', 'cls', 'dfl'} or any(not 0 < float(v) < float('inf') for v in gains.values()):
            raise ValueError('Explicit finite positive box/cls/dfl gains required')
        self.native.args = SimpleNamespace(**gains)
        self.construction_receipt = dict(family=family, seed=seed, gains=gains,
                                         ultralytics=ultralytics.__version__)
        self._criterion = None

    def _native_forward(self, images):
        """Single override seam for adapters that preserve the native YOLO graph."""
        return self.native(images)

    def load_official_coco_tuning(self, checkpoint_path, expected_sha256=None):
        """Strictly transfer official COCO weights except the native class branch."""
        path = Path(checkpoint_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        checksum = _file_sha256(path)
        if expected_sha256 is not None and checksum != str(expected_sha256).lower():
            raise ValueError('Official YOLO checkpoint SHA-256 differs from frozen value')
        checkpoint = torch.load(path, map_location='cpu')
        source, branch = None, None
        for name in ('ema', 'model'):
            candidate = checkpoint.get(name) if isinstance(checkpoint, dict) else None
            if isinstance(candidate, torch.nn.Module):
                source, branch = candidate.float().state_dict(), name
                break
            if isinstance(candidate, dict) and candidate:
                source, branch = candidate, name
                break
        if source is None and isinstance(checkpoint, dict) and isinstance(
                checkpoint.get('model_state'), dict):
            source, branch = checkpoint['model_state'], 'model_state'
        if not source or not all(isinstance(key, str) and torch.is_tensor(value)
                                 for key, value in source.items()):
            raise ValueError('Expected official YOLO model or tensor state')

        target = self.native.state_dict()
        class_prefix = f'model.{self.head_index}.cv3.'
        matched, allowed, forbidden = {}, [], []
        for key, target_value in target.items():
            source_value = source.get(key)
            if source_value is not None and source_value.shape == target_value.shape:
                matched[key] = source_value
            elif key.startswith(class_prefix):
                allowed.append(key)
            else:
                reason = 'missing' if source_value is None else (
                    f'shape {tuple(source_value.shape)} != {tuple(target_value.shape)}')
                forbidden.append(f'{key}: {reason}')
        if forbidden:
            raise ValueError('Official YOLO checkpoint is incompatible outside class branch: '
                             + '; '.join(forbidden[:12]))
        incompatible = self.native.load_state_dict(matched, strict=False)
        if set(incompatible.missing_keys) != set(allowed) or incompatible.unexpected_keys:
            raise RuntimeError('Audited YOLO COCO initialization changed during load')
        eligible_numel = sum(value.numel() for key, value in target.items()
                              if not key.startswith(class_prefix))
        matched_numel = sum(target[key].numel() for key in matched
                             if not key.startswith(class_prefix))
        if matched_numel != eligible_numel:
            raise RuntimeError('Not all eligible YOLO COCO parameters were initialized')
        receipt = dict(
            checkpoint_path=str(path), checkpoint_size=path.stat().st_size,
            checkpoint_sha256=checksum, source_branch=branch,
            matched_target_keys=sorted(matched), matched_eligible_numel=matched_numel,
            eligible_target_numel=eligible_numel,
            allowed_unmatched_target_keys=sorted(allowed),
            ignored_source_keys=sorted(set(source) - set(target)),
        )
        self.construction_receipt['official_coco_initialization'] = receipt
        return receipt

    def forward(self, inputs, mode='loss'):
        if mode == 'predict':
            if self.training or set(inputs) != {'img', 'metadata'}:
                raise ValueError('Prediction requires eval mode and label-free inputs')
            from maize_detector.prediction import decode_yolo
            decoded = self._native_forward(inputs['img'])[0]
            return decode_yolo(decoded, inputs['metadata'], tuple(inputs['img'].shape[-2:]))
        if mode != 'loss':
            raise ValueError('mode must be loss or predict')
        if not self.training:
            raise ValueError('Loss adapter requires native training-mode raw logits')
        device = next(self.native.parameters()).device
        if self._criterion is None or self._criterion.device != device:
            self._criterion = SupervisedYoloLoss(self.native)
        values, _ = self._criterion(self._native_forward(inputs['img']), inputs)
        return dict(loss_box=values[0], loss_cls=values[1], loss_dfl=values[2])


class MaizeDetector(YoloDetector):
    """RCFF-YOLO-n and RCFF-YOLO-s; the head and neck have separate conditioners."""

    def __init__(self, scale='s', seed=0, gains=None, *, rpg=True, fusion=True):
        if scale not in ('n', 's'):
            raise ValueError('scale must be n or s')
        family = scale.upper() + '11'
        super().__init__(family=family, seed=seed, gains=gains)
        self.scale = scale
        self.feature_gate_enabled = bool(rpg)
        self.frequency_fusion_enabled = bool(fusion)
        if rpg:
            self.perspective_gate = RowConditioner(state_id=seed)
        if fusion:
            self.high_channels, self.low_channels = self._verify_fusion_seam(family)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(int(seed) + 240924)
                self.frequency_fusion = upstream_module().FreqFusion(
                    hr_channels=self.high_channels, lr_channels=self.low_channels,
                    compressed_channels=64, lowpass_kernel=5, highpass_kernel=3,
                    feature_resample=True, feature_resample_group=4)
            self.frequency_fusion.float()
            self.row_neck_conditioner = RowConditioner(state_id=seed)

    @property
    def neck_conditioner(self):
        return self.row_neck_conditioner

    def _native_forward(self, images):
        handles = []
        try:
            if self.frequency_fusion_enabled:
                handles.append(self.native.model[14].register_forward_hook(self._bypass_upsample))
                handles.append(self.native.model[15].register_forward_pre_hook(self._fuse_inputs))
            if self.feature_gate_enabled:
                def gate(_module, args):
                    return (list(self.perspective_gate(args[0])),)
                handles.append(self.native.model[-1].register_forward_pre_hook(gate))
            return self.native(images)
        finally:
            for handle in handles:
                handle.remove()


    def _verify_fusion_seam(self, family):
        layers = self.native.model
        if (len(layers) != 24 or not isinstance(layers[14], torch.nn.Upsample)
                or layers[15].__class__.__name__ != 'Concat' or layers[15].f != [-1, 4]):
            raise ValueError('YOLO11 neck topology changed; refusing to guess the fusion seam')
        upsample = layers[14]
        if (upsample.mode != 'nearest' or upsample.size is not None
                or upsample.scale_factor != 2.):
            raise ValueError('The verified native fusion seam requires nearest 2x upsampling')
        try:
            high = layers[4].cv2.conv.out_channels
            low = layers[13].cv2.conv.out_channels
        except AttributeError as error:
            raise ValueError('YOLO11 channel seam changed') from error
        expected = 256 if family == 'S11' else 128
        if high != expected or low != expected:
            raise ValueError(f'{family} fusion branches must both have {expected} channels')
        return high, low

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        for name in ('frequency_fusion',):
            module = getattr(self, name, None)
            if module is not None:
                module.float()
        return result

    @staticmethod
    def _bypass_upsample(_module, arguments, _output):
        if len(arguments) != 1 or not isinstance(arguments[0], torch.Tensor):
            raise ValueError('Native Upsample input seam changed')
        return arguments[0]

    def _fuse_inputs(self, _module, arguments):
        if (len(arguments) != 1 or not isinstance(arguments[0], (list, tuple))
                or len(arguments[0]) != 2):
            raise ValueError('Native Concat must receive [low, high]')
        low, high = arguments[0]
        if not all(isinstance(branch, torch.Tensor) and branch.ndim == 4
                   and branch.is_floating_point() for branch in (low, high)):
            raise ValueError('Fusion branches must be floating BCHW tensors')
        if (high.shape[0] != low.shape[0] or high.device != low.device
                or high.dtype != low.dtype or high.shape[1] != self.high_channels
                or low.shape[1] != self.low_channels
                or high.shape[-2:] != (2 * low.shape[-2], 2 * low.shape[-1])):
            raise ValueError('FreqFusion requires the verified channels and 2:1 feature resolution')
        dtype = high.dtype
        with torch.autocast(device_type=high.device.type, enabled=False):
            _, high_frequency, low_frequency = self.frequency_fusion(
                hr_feat=high.float(), lr_feat=low.float())
            if high_frequency.shape != high.shape or low_frequency.shape != (
                    low.shape[0], self.low_channels, *high.shape[-2:]):
                raise ValueError('Upstream frequency outputs changed shape')
            native_low = F.interpolate(low.float(), size=high.shape[-2:], mode='nearest')
            high_frequency, low_frequency = self.neck_conditioner.mix_residual(
                high.float(), native_low, high_frequency, low_frequency,
                original_low=low.float())
            return ([low_frequency.to(dtype), high_frequency.to(dtype)],)
