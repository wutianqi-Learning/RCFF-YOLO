# Third-party code and attribution

## Ultralytics

The architecture, detection losses, task-aligned assignment and NMS use [Ultralytics 8.3.160](https://github.com/ultralytics/ultralytics/tree/v8.3.160), distributed under [AGPL-3.0](licenses/Ultralytics-AGPL-3.0.txt). `maize_detector/loss.py` adapts the preparation and reduction logic of `v8DetectionLoss`. That derived logic retains the upstream AGPL-3.0 terms.

## FreqFusion

The frequency module comes from [Frequency-aware Feature Fusion for Dense Image Prediction](https://github.com/Linwei-Chen/FreqFusion), by its original authors. The download script pins commit `3fb0c70637a3c194fb74294d3ce4681958b26241` and SHA-256 `5852efce406681ce6f3f7763ff8e034cab44f93cb0fdda0114130bc4c8c97e42`.

The upstream file is downloaded separately and remains unmodified. Its implementation is not attributed to this repository. Consult the upstream repository for citation and applicable reuse terms.

## Other dependencies

- [MMCV](https://github.com/open-mmlab/mmcv) provides CUDA CARAFE.
- [PyTorch](https://github.com/pytorch/pytorch) provides tensor computation and automatic differentiation.
- [COCO API](https://github.com/cocodataset/cocoapi) provides bounding-box AP evaluation.

Each dependency retains its own license. No additional license grant for the original project code is specified in this release.
