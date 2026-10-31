"""Configuration, checkpoint loading and label-free inference."""
from contextlib import nullcontext
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import yaml
from .geometry import LetterboxGeometry


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_config(path):
    with Path(path).open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    return config


def build_model(config):
    from .model import MaizeDetector, YoloDetector
    model = config['model']
    gains = config.get('loss', dict(box=7.5, cls=.5, dfl=1.5))
    if model['name'] == 'ours':
        return MaizeDetector(scale=model['scale'], seed=config['seed'], gains=gains,
                             rpg=model.get('rpg', True), fusion=model.get('fusion', True))
    if model['name'] == 'yolo':
        return YoloDetector(family=model['family'], seed=config['seed'], gains=gains)
    raise ValueError('model.name must be ours or yolo')


def load_model(config, weights, device):
    model = build_model(config)
    # Release checkpoints contain plain tensor dictionaries. Use trusted local files.
    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    state = checkpoint.get('model', checkpoint)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def precision_context(device, precision):
    device = torch.device(device)
    if precision == 'fp32':
        return nullcontext()
    if precision != 'bf16' or device.type != 'cuda':
        raise ValueError('bf16 inference requires CUDA; use fp32 for CPU checks')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('This CUDA device does not support BF16')
    return torch.autocast('cuda', dtype=torch.bfloat16)


def predict_coco(model, coco, image_root, device, image_size=1280, precision='bf16'):
    predictions = []
    image_root = Path(image_root)
    for record in coco['images']:
        geometry = LetterboxGeometry.from_source(record['width'], record['height'], image_size)
        with Image.open(image_root / record['file_name']) as image:
            if image.size != (record['width'], record['height']):
                raise ValueError('Image size differs from the annotation record')
            pixels = np.array(geometry.apply_pil(image.convert('RGB')), copy=True)
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=device, dtype=torch.float32).div_(255.)
        metadata = dict(id=record['id'], width=record['width'], height=record['height'],
                        preprocessing=geometry.as_dict())
        with torch.inference_mode(), precision_context(device, precision):
            result = model({'img': tensor, 'metadata': [metadata]}, mode='predict')[0]
        instances = result['pred_instances']
        for box, score in zip(instances['bboxes'].detach().cpu().tolist(),
                              instances['scores'].detach().cpu().tolist()):
            x1, y1, x2, y2 = box
            x, y = max(0., min(float(x1), record['width'])), max(0., min(float(y1), record['height']))
            w = min(float(x2) - x, record['width'] - x)
            h = min(float(y2) - y, record['height'] - y)
            if w > 0 and h > 0:
                predictions.append(dict(image_id=record['id'], category_id=1,
                                        bbox=[x, y, w, h], score=float(score)))
    return predictions


def evaluate(model, coco, image_root, device, image_size=1280, precision='bf16'):
    from .evaluation import evaluate_coco_partition, evaluate_working_point
    predictions = predict_coco(model, coco, image_root, device, image_size, precision)
    ap = evaluate_coco_partition(coco, predictions, canvas_size=image_size)
    working = evaluate_working_point(coco, predictions, threshold=.25, iou=.5)
    return dict(ap=ap['ap'], ap50=ap['ap50'], ap75=ap['ap75'], **working), predictions
