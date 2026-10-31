"""Frozen native decoding into original pixels; no annotation/ignore inputs."""
import torch

SCORE_FLOOR = .001
MAX_DETECTIONS = 100


def _canvas_to_original(boxes, metadata, canvas_size):
    """Invert the declared preprocessing without moving tensors off device."""
    payload = metadata.get('preprocessing')
    if payload is None:
        height, width = canvas_size
        return boxes * boxes.new_tensor([
            metadata['width'] / width, metadata['height'] / height,
            metadata['width'] / width, metadata['height'] / height])
    from maize_detector.geometry import LetterboxGeometry

    geometry = LetterboxGeometry.from_dict(payload)
    if tuple(canvas_size) != (geometry.canvas_size, geometry.canvas_size):
        raise ValueError('Prediction canvas differs from declared preprocessing')
    result = boxes.clone()
    result[:, 0::2] = (result[:, 0::2] - geometry.pad_left) / geometry.scale_x
    result[:, 1::2] = (result[:, 1::2] - geometry.pad_top) / geometry.scale_y
    return result


def prediction_record(boxes, scores, labels, metadata):
    if (boxes.shape != (len(scores), 4) or labels.shape != scores.shape
            or not torch.isfinite(boxes).all() or not torch.isfinite(scores).all()
            or (labels != 0).any() or (scores < 0).any() or (scores > 1).any()):
        raise ValueError('Invalid single-class original-pixel predictions')
    width, height = metadata['width'], metadata['height']
    boxes = boxes.float().clone()
    boxes[:, 0::2].clamp_(0, width)
    boxes[:, 1::2].clamp_(0, height)
    keep = (scores > SCORE_FLOOR) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    order = scores[keep].argsort(descending=True, stable=True)[:MAX_DETECTIONS]
    return dict(img_id=metadata['id'], ori_shape=(height, width),
        pred_instances=dict(bboxes=boxes[keep][order], scores=scores[keep][order].float(),
                            labels=labels[keep][order].long()))


def decode_yolo(decoded, metadata, canvas_size):
    from ultralytics.utils.ops import non_max_suppression
    # Validation batch1: native wall-time break cannot hide unprocessed tail rows.
    if len(metadata) != 1 or len(decoded) != 1:
        raise ValueError('Frozen validation uses batch1')
    predictions = non_max_suppression(decoded, conf_thres=SCORE_FLOOR, iou_thres=.7,
        nc=1, multi_label=False, agnostic=False, labels=(), max_det=MAX_DETECTIONS,
        max_nms=30000, in_place=False, end2end=False)
    h, w = canvas_size
    result = []
    for pred, record in zip(predictions, metadata):
        boxes = _canvas_to_original(pred[:, :4], record, (h, w))
        result.append(prediction_record(boxes, pred[:, 4], pred[:, 5], record))
    return result
