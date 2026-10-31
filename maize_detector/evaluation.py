"""COCO box AP and fixed-threshold precision and recall."""
import contextlib
import copy
import io
import math
import numpy as np
from .geometry import LetterboxGeometry

PARTITIONS = ('all', 'far_like', 'mid_like', 'near_like')
PREDICTION_FLOOR = .001
MAX_DETECTIONS = 100

def _valid_coco(coco):
    if (not isinstance(coco, dict)
            or coco.get('categories') != [dict(id=1, name='plant')]
            or not isinstance(coco.get('images'), list)
            or not isinstance(coco.get('annotations'), list)):
        raise ValueError('Expected canonical single-class COCO data')
    images = {row.get('id'): row for row in coco['images']}
    if len(images) != len(coco['images']) or None in images:
        raise ValueError('Evaluation images require unique non-null IDs')
    file_names = set()
    for image in images.values():
        name = image.get('file_name')
        if (not isinstance(name, str) or not name or name in file_names
                or type(image.get('width')) is not int
                or type(image.get('height')) is not int
                or min(image['width'], image['height']) < 1):
            raise ValueError('Evaluation images require unique names and valid shapes')
        file_names.add(name)
    annotation_ids = set()
    for annotation in coco['annotations']:
        identifier = annotation.get('id')
        bbox = annotation.get('bbox')
        if (identifier is None or identifier in annotation_ids
                or annotation.get('image_id') not in images
                or annotation.get('category_id') != 1
                or not isinstance(bbox, list) or len(bbox) != 4
                or not all(math.isfinite(float(value)) for value in bbox)
                or bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= 0 or bbox[3] <= 0
                or bbox[0] + bbox[2] > images[annotation['image_id']]['width']
                or bbox[1] + bbox[3] > images[annotation['image_id']]['height']
                or not math.isfinite(float(annotation.get('area', float('nan'))))
                or not math.isclose(float(annotation['area']),
                                    float(bbox[2]) * float(bbox[3]),
                                    rel_tol=1e-6, abs_tol=1e-6)
                or annotation.get('iscrowd', 0) not in (0, 1)):
            raise ValueError('Invalid evaluation annotation')
        annotation_ids.add(identifier)
    return images


def _partition(bbox, image, name, canvas_size):
    if name == 'all':
        return True
    geometry = LetterboxGeometry.from_source(
        int(image['width']), int(image['height']), int(canvas_size))
    return geometry.labels_for_source_xywh(bbox)[name]


def _prediction_rows(predictions, images):
    rows = []
    for row in predictions:
        if (not isinstance(row, dict) or row.get('image_id') not in images
                or row.get('category_id') != 1
                or not isinstance(row.get('score'), (int, float))
                or not 0 <= float(row['score']) <= 1):
            raise ValueError('Invalid or foreign prediction row')
        bbox = row.get('bbox')
        if (not isinstance(bbox, list) or len(bbox) != 4
                or not all(math.isfinite(float(value)) for value in bbox)
                or bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= 0 or bbox[3] <= 0
                or bbox[0] + bbox[2] > images[row['image_id']]['width']
                or bbox[1] + bbox[3] > images[row['image_id']]['height']):
            raise ValueError('Predictions require finite positive COCO xywh boxes')
        rows.append(dict(row, score=float(row['score'])))
    return rows


def _partition_view(coco, predictions, partition, canvas_size):
    if partition not in PARTITIONS:
        raise ValueError(f'Unknown partition: {partition}')
    images = _valid_coco(coco)
    predictions = _prediction_rows(predictions, images)
    view = copy.deepcopy(coco)
    target_ids = set()
    for annotation in view['annotations']:
        is_target = (annotation.get('iscrowd', 0) == 0
                     and _partition(annotation['bbox'], images[annotation['image_id']],
                                    partition, canvas_size))
        if is_target:
            target_ids.add(annotation['id'])
            annotation['ignore'] = 0
        else:
            # Non-target plants and explicit ignore regions must absorb matching
            # detections rather than turn correct predictions at other scales into FP.
            annotation['ignore'] = 1
            annotation['iscrowd'] = 1
    filtered = [row for row in predictions
                if _partition(row['bbox'], images[row['image_id']],
                              partition, canvas_size)]
    return view, filtered, len(target_ids)


def evaluate_coco_partition(coco, predictions, partition='all', *, canvas_size=1280,
                            max_detections=MAX_DETECTIONS,
                            prediction_floor=PREDICTION_FLOOR):
    """Return native COCO AP for one observable geometric partition."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if (not 0 <= float(prediction_floor) <= 1
            or type(max_detections) is not int or max_detections < 1):
        raise ValueError('Invalid prediction floor or maximum detections')
    predictions = [row for row in predictions
                   if float(row.get('score', -1)) >= prediction_floor]
    view, filtered, target_count = _partition_view(
        coco, predictions, partition, canvas_size)
    if target_count == 0:
        return dict(partition=partition, gt=0, predictions=len(filtered),
                    ap=-1., ap50=-1., ap75=-1.,
                    prediction_floor=float(prediction_floor),
                    max_detections=int(max_detections))
    ground_truth = COCO()
    ground_truth.dataset = view
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth.createIndex()
        detections = ground_truth.loadRes(filtered) if filtered else COCO()
        if not filtered:
            detections.dataset = dict(images=view['images'],
                                      categories=view['categories'], annotations=[])
            detections.createIndex()
        evaluator = COCOeval(ground_truth, detections, 'bbox')
        evaluator.params.imgIds = sorted(ground_truth.getImgIds())
        evaluator.params.catIds = [1]
        evaluator.params.maxDets = [1, 10, int(max_detections)]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {
        'partition': partition,
        'gt': target_count,
        'predictions': len(filtered),
        'ap': float(evaluator.stats[0]),
        'ap50': float(evaluator.stats[1]),
        'ap75': float(evaluator.stats[2]),
        'prediction_floor': float(prediction_floor),
        'max_detections': int(max_detections),
    }


def _xywh_to_xyxy(bbox):
    x, y, width, height = (float(value) for value in bbox)
    return np.asarray([x, y, x + width, y + height], dtype=np.float64)


def _iou(box, boxes, crowd=False):
    if not len(boxes):
        return np.empty(0, dtype=np.float64)
    left_top = np.maximum(box[:2], boxes[:, :2])
    right_bottom = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.maximum(0., right_bottom - left_top).prod(axis=1)
    box_area = np.maximum(0., box[2:] - box[:2]).prod()
    other_area = np.maximum(0., boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    denominator = box_area if crowd else box_area + other_area - intersection
    return np.divide(intersection, denominator, out=np.zeros_like(intersection),
                     where=denominator > 0)


def _counts_by_group(coco, predictions, threshold, iou):
    images = _valid_coco(coco)
    rows = _prediction_rows(predictions, images)
    annotations = {identifier: [] for identifier in images}
    detections = {identifier: [] for identifier in images}
    for annotation in coco['annotations']:
        annotations[annotation['image_id']].append(annotation)
    for row in rows:
        if row['score'] >= threshold:
            detections[row['image_id']].append(row)
    counts = {row['group_id']: [0, 0, 0] for row in images.values()}
    for identifier, image in images.items():
        group = image['group_id']
        positives = [_xywh_to_xyxy(row['bbox']) for row in annotations[identifier]
                     if row.get('iscrowd', 0) == 0]
        ignored = [_xywh_to_xyxy(row['bbox']) for row in annotations[identifier]
                   if row.get('iscrowd', 0) == 1]
        positives = np.asarray(positives, dtype=np.float64).reshape(-1, 4)
        ignored = np.asarray(ignored, dtype=np.float64).reshape(-1, 4)
        matched = np.zeros(len(positives), dtype=bool)
        ordered = sorted(detections[identifier], key=lambda row: row['score'],
                         reverse=True)[:100]
        for row in ordered:
            box = _xywh_to_xyxy(row['bbox'])
            overlaps = _iou(box, positives)
            overlaps[matched] = -1
            best = int(overlaps.argmax()) if len(overlaps) else -1
            if best >= 0 and overlaps[best] >= iou:
                matched[best] = True
                counts[group][0] += 1
            elif len(ignored) and bool((_iou(box, ignored, crowd=True) >= iou).any()):
                continue
            else:
                counts[group][1] += 1
        counts[group][2] += int((~matched).sum())
    return counts


def _prf(counts):
    tp, fp, fn = (sum(value[index] for value in counts.values())
                  for index in range(3))
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    group_f1 = []
    for group_tp, group_fp, group_fn in counts.values():
        denominator = 2 * group_tp + group_fp + group_fn
        group_f1.append(2 * group_tp / denominator if denominator else 0.)
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1,
                macro_group_f1=float(np.mean(group_f1)))


def evaluate_working_point(coco, predictions, *, threshold, iou=.5):
    if not 0 <= float(threshold) <= 1 or not 0 < float(iou) <= 1:
        raise ValueError('Invalid frozen threshold or IoU')
    values = _prf(_counts_by_group(coco, predictions, float(threshold), float(iou)))
    return dict(threshold=float(threshold), iou=float(iou), **values)
