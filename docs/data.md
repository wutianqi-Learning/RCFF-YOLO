# Data format

The loader accepts COCO bounding boxes in original image pixels. The single category is `plant` with ID `1`. Image IDs and annotation IDs must be unique within each file. `file_name` is relative to `image_root`.

## Example

```json
{
  "images": [{
    "id": 1, "file_name": "video_01/frame_0001.jpg",
    "width": 1920, "height": 1080,
    "group_id": "video_01", "split": "train",
    "annotation_origin": "official_nearfar_mixed_manual_propagated"
  }],
  "annotations": [{
    "id": 1, "image_id": 1, "category_id": 1,
    "bbox": [600, 420, 80, 120], "area": 9600, "iscrowd": 0
  }],
  "categories": [{"id": 1, "name": "plant"}]
}
```

This example illustrates the schema; the box and filename are not dataset annotations.

## Splits and labels

Use `train`, `validation` or `test` in the `split` field. All frames from a video group belong to the same split. Training checks that training and validation groups and filenames are disjoint.

Use `annotation_origin: human` for manual validation and test annotations. MaizeHorizon nearfar training uses `official_nearfar_mixed_manual_propagated`. These labels receive full positive/background supervision outside ignore regions. Internal loss compatibility fields do not change the stored provenance.

For an additional dataset with positive-only pseudo labels, explicitly set these image fields.

```json
{
  "annotation_origin": "pseudo",
  "maize_annotation_origin": "pseudo",
  "maize_loss_weight": 0.5,
  "maize_supervision_policy": "positive_only_rest_unknown"
}
```

This supervises positive assignments and leaves unannotated background unknown. It differs from full supervision with nearfar annotations.

## Ignore regions and geometry

Use `iscrowd: 1` for ignore boxes and `iscrowd: 0` for plant boxes. `area` equals width multiplied by height. Boxes must stay within image bounds. Training applies the same letterbox and flip to positive and ignore boxes.

P/R matches predictions to positive boxes first. An unmatched prediction covered by an ignore region over at least half its area is excluded from false positives. AP uses the COCO crowd convention.

For an existing experiment, retain the original image membership and video-group assignments. Randomly resplitting adjacent frames changes the evaluation. The supplied recipe used 3,349 MaizeHorizon nearfar training frames. Images and annotations are not distributed in this code repository.
