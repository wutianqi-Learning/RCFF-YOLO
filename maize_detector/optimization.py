"""AdamW grouping, update-wise scheduling, accumulation and EMA."""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import tempfile

TOTAL_UPDATES = 41900
WARMUP_UPDATES = 1257
PEAK_LR = 3e-4
WEIGHT_DECAY = 1e-4
ACCUMULATION = 8

def learning_rate(update, *, total_updates=TOTAL_UPDATES,
                  warmup_updates=WARMUP_UPDATES, initial=PEAK_LR,
                  final_factor=.01):
    """One-based optimizer-update schedule; the first update is .001*peak."""
    if (type(update) is not int or type(total_updates) is not int
            or type(warmup_updates) is not int
            or not 1 <= update <= total_updates
            or not 1 <= warmup_updates < total_updates
            or not math.isfinite(initial) or not initial > 0
            or not math.isfinite(final_factor) or not 0 < final_factor <= 1):
        raise ValueError('Invalid fixed learning-rate schedule parameters')
    if update <= warmup_updates:
        progress = (update - 1) / max(1, warmup_updates - 1)
        return initial * (.001 + .999 * progress)
    progress = (update - warmup_updates) / (total_updates - warmup_updates)
    return initial * (final_factor + .5 * (1 - final_factor)
                      * (1 + math.cos(math.pi * progress)))


def accumulation_divisor(batch_index, batch_count, accumulation=ACCUMULATION):
    """Normalize each microbatch by its actual (possibly partial) group size."""
    if (type(batch_index) is not int or type(batch_count) is not int
            or type(accumulation) is not int
            or not 0 <= batch_index < batch_count or accumulation < 1):
        raise ValueError('Invalid accumulation group')
    start = batch_index // accumulation * accumulation
    return min(accumulation, batch_count - start)


def selection_key(metrics):
    """Use unrounded same-checkpoint AP50:95, AP50, then F1, in that order."""
    values = tuple(metrics[key] for key in ('ap', 'ap50', 'f1'))
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in values):
        raise ValueError('Checkpoint selection requires finite numeric metrics')
    return values


def ema_decay(updates):
    if type(updates) is not int or updates < 1:
        raise ValueError('EMA decay requires a positive optimizer-update count')
    return .9999 * (1 - math.exp(-updates / 2000))


def atomic_json(path, value):
    """Atomically replace an owned rolling JSON artifact, rejecting NaN/Inf."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         prefix=path.name + '.', suffix='.partial',
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def parameter_groups(model, weight_decay=WEIGHT_DECAY):
    """Group AdamW parameters, excluding biases and normalization from decay.

    Only tensors named ``weight`` receive decay. If a parameter is shared,
    a no-decay assignment takes precedence over a decay assignment.
    """
    import torch

    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError('Finite nonnegative weight decay required')
    normalization_types = (
        torch.nn.modules.batchnorm._NormBase,
        torch.nn.modules.instancenorm._InstanceNorm,
        torch.nn.LayerNorm, torch.nn.GroupNorm,
        torch.nn.LocalResponseNorm,
    )
    if hasattr(torch.nn, 'RMSNorm'):
        normalization_types += (torch.nn.RMSNorm,)
    named = [(name, value) for name, value in model.named_parameters()
             if value.requires_grad]
    expected = {id(value) for _, value in named}
    if not expected or len(expected) != len(named):
        raise RuntimeError('Trainable parameter names must identify unique parameters')
    no_decay = set()
    seen = set()
    for _, module in model.named_modules():
        is_norm = isinstance(module, normalization_types)
        for local_name, value in module.named_parameters(recurse=False):
            if value.requires_grad:
                seen.add(id(value))
                if is_norm or local_name != 'weight':
                    no_decay.add(id(value))
    if seen != expected:
        raise RuntimeError('Module traversal does not cover all trainable parameters')
    decay_named = [(name, value) for name, value in named if id(value) not in no_decay]
    plain_named = [(name, value) for name, value in named if id(value) in no_decay]
    groups = []
    for values, decay in ((decay_named, weight_decay), (plain_named, 0.0)):
        if values:
            groups.append(dict(params=[value for _, value in values], weight_decay=decay))
    actual = [id(value) for group in groups for value in group['params']]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise RuntimeError('AdamW groups duplicate or omit trainable parameters')
    receipt = dict(
        trainable_tensors=len(named), trainable_parameters=sum(p.numel() for _, p in named),
        decay_names=[name for name, _ in decay_named],
        no_decay_names=[name for name, _ in plain_named],
        weight_decay=weight_decay, unique_identity_coverage=True,
    )
    return groups, receipt


class EMAState:
    """FP32 floating shadow state, with non-floating buffers copied verbatim."""

    def __init__(self, model):
        self.updates = 0
        self.shadow = {
            key: value.detach().float().clone() if value.is_floating_point()
            else value.detach().clone()
            for key, value in model.state_dict().items()
        }

    def update(self, model, updates):
        import torch

        if type(updates) is not int or updates != self.updates + 1:
            raise ValueError('EMA must update exactly once per optimizer step')
        state = model.state_dict()
        if set(state) != set(self.shadow):
            raise RuntimeError('Model state keys changed after EMA initialization')
        decay = ema_decay(updates)
        with torch.no_grad():
            finite = [torch.isfinite(value).all() for value in state.values()
                      if value.is_floating_point()]
            if finite and not torch.stack(finite).all():
                raise FloatingPointError('Nonfinite model state after optimizer step')
            for key, value in state.items():
                target = self.shadow[key]
                if target.shape != value.shape or target.device != value.device:
                    raise RuntimeError('EMA state shape or device changed: ' + key)
                if value.is_floating_point():
                    target.mul_(decay).add_(value.detach().float(), alpha=1 - decay)
                else:
                    target.copy_(value.detach())
        self.updates = updates

    def state_dict(self, *, cpu=False):
        return dict(updates=self.updates, shadow={
            key: (value.detach().cpu().clone() if cpu else value.detach().clone())
            for key, value in self.shadow.items()
        })


@contextmanager
def apply_ema(model, ema):
    """Temporarily install EMA; restore raw buffers and training flags on errors."""
    raw = {key: value.detach().cpu().clone()
           for key, value in model.state_dict().items()}
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.load_state_dict(ema.shadow, strict=True)
        yield model
    finally:
        model.load_state_dict(raw, strict=True)
        for module, training in modes:
            module.training = training
