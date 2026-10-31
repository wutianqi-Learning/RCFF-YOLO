"""COCO training data with centered letterbox and horizontal flips."""
import copy
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from .supervision import ImageSupervision

class CocoTrainingDataset(Dataset):
    """Centered letterbox plus shared horizontal flip of GT and ignore.

    COCO annotations contain explicit split and label-provenance fields.  Target
    coordinates normalize to the final square canvas; ignore stays in final
    pixel xyxy.  The exact geometry is attached to metadata for prediction
    inversion and geometric evaluation strata.
    """
    def __init__(self, coco_path, image_root, image_size=1280, flip_probability=.5):
        if image_size < 64 or image_size % 32 or not 0 <= flip_probability <= 1:
            raise ValueError('Size must be >=64, multiple32; flip probability in [0,1]')
        self.image_size, self.flip_probability = image_size, flip_probability
        self.root = Path(image_root)
        coco = json.loads(Path(coco_path).read_text())
        if [(x['id'], x['name']) for x in coco['categories']] != [(1, 'plant')]:
            raise ValueError('Expected canonical single COCO plant category1')
        self.images = coco['images']
        self.annotations = {row['id']: [] for row in self.images}
        if len(self.annotations) != len(self.images):
            raise ValueError('Duplicate image IDs')
        for row in self.images:
            # Official nearfar annotations use full supervision, including propagated boxes.
            # Preserve their provenance and set only the internal loss-policy fields.
            if row.get('annotation_origin') == 'official_nearfar_mixed_manual_propagated':
                row.update(maize_annotation_origin='human', maize_loss_weight=1.,
                           maize_supervision_policy='human_positive_and_background_except_ignore')
            elif 'maize_annotation_origin' not in row:
                if row.get('annotation_origin') != 'human':
                    raise ValueError('Set annotation origin and an explicit supervision policy')
                row.update(maize_annotation_origin='human', maize_loss_weight=1.,
                           maize_supervision_policy='human_positive_and_background_except_ignore')
            ImageSupervision.from_coco_image(row)
            if row['split'] != 'train':
                raise ValueError('Training Dataset refuses nontrain images')
        for ann in coco['annotations']:
            if ann['category_id'] != 1 or ann.get('iscrowd', 0) not in (0, 1):
                raise ValueError('Unsupported canonical annotation')
            self.annotations[ann['image_id']].append(ann)

    def __len__(self):
        return len(self.images)

    def get_data_info(self, index):
        return copy.deepcopy(self.images[index])

    def __getitem__(self, index):
        from maize_detector.geometry import LetterboxGeometry

        record = self.get_data_info(index)
        geometry = LetterboxGeometry.from_source(
            int(record['width']), int(record['height']), int(self.image_size))
        with Image.open(self.root / record['file_name']) as image:
            if image.size != (record['width'], record['height']):
                raise ValueError('Actual image dimensions differ from verified COCO')
            image = geometry.apply_pil(image.convert('RGB'))
            pixels = np.array(image, dtype=np.uint8, copy=True)
        flip = random.random() < self.flip_probability
        if flip:
            pixels = pixels[:, ::-1].copy()
        positive, ignored = [], []
        width, height = record['width'], record['height']
        for ann in self.annotations[record['id']]:
            x, y, w, h = ann['bbox']
            source_box = np.asarray([[x, y, x + w, y + h]], dtype=np.float64)
            if (not np.isfinite(source_box).all() or not
                    (0 <= x < x + w <= width and 0 <= y < y + h <= height)):
                raise ValueError('Invalid canonical box geometry')
            box = geometry.to_canvas_xyxy(source_box)[0] / self.image_size
            if flip:
                box = box[[2, 1, 0, 3]] * [-1, 1, -1, 1] + [1, 0, 1, 0]
            (ignored if ann.get('iscrowd', 0) else positive).append(box.tolist())
        xyxy = torch.tensor(positive, dtype=torch.float32).reshape(-1, 4)
        cxcywh = torch.cat(((xyxy[:, :2] + xyxy[:, 2:]) / 2, xyxy[:, 2:] - xyxy[:, :2]), 1)
        record['preprocessing'] = geometry.as_dict()
        record['augmentation'] = dict(
            resize_policy='center_letterbox_v1', square_size=self.image_size,
            horizontal_flip=flip, scale_x=geometry.scale_x,
            scale_y=geometry.scale_y, pad_left=geometry.pad_left,
            pad_top=geometry.pad_top)
        return dict(img=torch.from_numpy(pixels).permute(2, 0, 1).float() / 255., bboxes=cxcywh,
                    ignored_xyxy=torch.tensor(ignored, dtype=torch.float32).reshape(-1, 4) * self.image_size,
                    metadata=record)


def collate_training(rows):
    """Stack images and retain per-image supervision and ignore regions."""
    if not rows:
        raise ValueError('Empty batch')
    boxes = torch.cat([row['bboxes'] for row in rows])
    return dict(inputs=dict(img=torch.stack([row['img'] for row in rows]), bboxes=boxes,
        cls=torch.zeros(len(boxes), 1), batch_idx=torch.cat([
            torch.full((len(row['bboxes']),), float(i)) for i, row in enumerate(rows)]),
        supervision=[copy.deepcopy(row['metadata']) for row in rows],
        ignored_xyxy=[row['ignored_xyxy'] for row in rows]))
