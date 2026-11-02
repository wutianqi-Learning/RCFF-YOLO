# RCFF-YOLO

[中文说明](README_zh.md)

Implementation of **Row-Conditioned Frequency Fusion** for the paper *Mitigating Perspective Scale Attenuation and Canopy Aliasing: Row-Conditioned Frequency Fusion for Maize Seedling Detection*.

This repository implements **RCFF-YOLO-n** and **RCFF-YOLO-s**, YOLO11 detectors for maize seedlings in forward-looking agricultural imagery. Row-wise Pyramid Gain (RPG) reweights the detection-scale features. Frequency-aware Feature Fusion (FreqFusion) refines the high-resolution neck features. Two independently parameterized row conditioners control these operations.

The release contains the model, training loop, evaluation code, inference commands and configurations. Datasets and trained weights are supplied separately.

## Model

The backbone and detection head follow YOLO11. A four-channel row descriptor combines feature magnitude, vertical variation, row position and pyramid level. A pointwise network maps this descriptor to bounded gains before the detection head. A separate network controls residual mixing between the native neck features and the FreqFusion outputs.

| Component | Implementation |
| --- | --- |
| Backbone | YOLO11n or YOLO11s, one plant category |
| RPG | Shared across P3–P5; gain `1 + 0.5 × tanh(z)` |
| Frequency fusion | Stride-8 neck node, adaptive filtering and learnable coordinate resampling |
| Residual mixing | Independent head/neck parameters; initial frequency contribution 0.1 |
| Precision | BF16 native network; FP32 conditioning and frequency operations |

See [model.py](maize_detector/model.py) and [conditioning.py](maize_detector/conditioning.py). The final model retains the FreqFusion coordinate sampler and the native detection head, task-aligned assignment, CIoU and distribution focal loss.

## Installation

Use Linux, Python 3.11 and an NVIDIA CUDA GPU. The reference environment uses PyTorch 2.1.0, CUDA 11.8, Ultralytics 8.3.160 and MMCV 2.1.0. MMCV must include its compiled CARAFE operator; `mmcv-lite` is insufficient.

Run these commands from the repository root in a new environment.

```bash
git clone https://github.com/wutianqi-Learning/RCFF-YOLO.git
cd RCFF-YOLO
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
python -m pip install -e . --no-deps
python scripts/fetch_freqfusion.py
```

The last command downloads the unmodified [upstream FreqFusion source](https://github.com/Linwei-Chen/FreqFusion) at a pinned revision and verifies its SHA-256 digest. Attribution and licensing information are in [THIRD_PARTY.md](THIRD_PARTY.md).

Check the installation with the unit tests and a small CUDA forward/backward pass.

```bash
python -m unittest discover -s tests -v
python scripts/smoke_model.py --scale n --device cuda:0
python scripts/smoke_model.py --scale s --device cuda:0
```

## Data

Training and evaluation consume COCO box annotations with one category named `plant`. Each image records its video group, split and annotation origin. The [data guide](docs/data.md) specifies these fields and the ignore-region convention.

The MaizeHorizon training recipe uses the public nearfar annotations, including manual and propagated labels. Validation and test evaluation use manual annotations from separate video groups. The supplied configuration does not mix in Maize-FV.

```bash
cp configs/data.example.yaml configs/data.local.yaml
```

Set `image_root`, `train` and `validation` to your local files. Keep test annotations outside the training configuration. Data, weights and local path configurations are ignored by Git.

## Training

Place official COCO-pretrained `yolo11s.pt` and `yolo11n.pt` files in `weights/`. Use weights from the [Ultralytics assets release](https://github.com/ultralytics/assets/releases/tag/v8.3.0). Compatible pretrained tensors are transferred, while unmatched class-prediction tensors are initialized for the plant category.

```bash
python -m maize_detector train \
  --config configs/rcff_yolo_s.yaml --data configs/data.local.yaml \
  --pretrained weights/yolo11s.pt \
  --output outputs/rcff_yolo_s_r1 --device cuda:0
```

For RCFF-YOLO-n, use `configs/rcff_yolo_n.yaml`, `weights/yolo11n.pt` and a separate output directory. Each run uses one GPU. Independent runs can be assigned to different GPUs through `--device` or `CUDA_VISIBLE_DEVICES`.

| Setting | Value |
| --- | --- |
| Duration | 100 epochs |
| Input | 1280 × 1280, centered letterbox |
| Optimizer | AdamW, betas 0.9 and 0.999, epsilon 1e-8 |
| Peak learning rate | 0.0003 |
| Weight decay | 0.0001; zero for normalization and bias parameters |
| Schedule | Three warmup epochs, then cosine decay to 1% of the peak |
| Batch | One image per forward pass, eight accumulation steps |
| Augmentation | Horizontal flip with probability 0.5 |
| Gradient clipping | Norm 10 |
| Weight selection | EMA; validation mAP50:95, then mAP50 and F1 as tie-breakers |
| Loss weights | Box 7.5, classification 0.5, DFL 1.5 |

The final partial accumulation group is normalized by its actual size. The loop saves `best.pt`, an epoch-boundary resume checkpoint `last.pt`, and per-epoch loss and validation metrics in `history.jsonl`.

Three independent runs use the following initialization states. Each run loads the same public COCO checkpoint.

```bash
for seed in 110022893 1970871798 1114612310; do
  python -m maize_detector train \
    --config configs/rcff_yolo_s.yaml --data configs/data.local.yaml \
    --pretrained weights/yolo11s.pt --seed "$seed" \
    --output "outputs/rcff_yolo_s_${seed}" --device cuda:0
done
```

Resume an interrupted run from a checkpoint created by this release.

```bash
python -m maize_detector train \
  --config configs/rcff_yolo_s.yaml --data configs/data.local.yaml \
  --resume outputs/rcff_yolo_s_r1/last.pt \
  --output outputs/rcff_yolo_s_r1 --device cuda:0
```

Keep the configuration, data paths, output directory and seed unchanged when resuming. Load checkpoints only from trusted sources.

## Evaluation

Evaluate a fixed checkpoint on a validation or test split.

```bash
python -m maize_detector evaluate \
  --config configs/rcff_yolo_s.yaml --weights outputs/rcff_yolo_s_r1/best.pt \
  --annotations data/maizehorizon/annotations/test.json \
  --image-root data/maizehorizon/images \
  --output outputs/rcff_yolo_s_r1/test --device cuda:0
```

The command writes `predictions.json` and `metrics.json`. AP uses `pycocotools` at IoU thresholds 0.50–0.95, a confidence floor of 0.001 and at most 100 detections per image. P/R use confidence 0.25 and IoU 0.50. NMS uses IoU 0.70. Ignore regions follow the crowd/overlap rules in the data guide.

JSON metrics are fractions; multiply by 100 for percentages. Evaluate all three runs and report their mean and sample standard deviation. Use validation results for selection and evaluate the held-out test set after fixing the model and configuration.

```bash
python scripts/summarize_runs.py \
  outputs/rcff_yolo_s_110022893/test/metrics.json \
  outputs/rcff_yolo_s_1970871798/test/metrics.json \
  outputs/rcff_yolo_s_1114612310/test/metrics.json
```

## Image inference

```bash
python -m maize_detector predict \
  --config configs/rcff_yolo_s.yaml --weights outputs/rcff_yolo_s_r1/best.pt \
  --image data/example.jpg --output outputs/example --device cuda:0
```

This saves a box visualization and COCO-format predictions. The display threshold defaults to 0.25 and can be changed with `--confidence`.

## Configurations

| Files | RPG | Conditional residual FreqFusion |
| --- | --- | --- |
| `baseline_n.yaml`, `baseline_s.yaml` | No | No |
| `rpg_n.yaml`, `rpg_s.yaml` | Yes | No |
| `freqfusion_n.yaml`, `freqfusion_s.yaml` | No | Yes |
| `rcff_yolo_n.yaml`, `rcff_yolo_s.yaml` | Yes | Yes, independent neck parameters |

All supplied configurations use the same 100-epoch recipe. Historical ablation budgets and alternative architectures are not reproduced by selecting these files alone. Change the `loss` fields to study loss balance factors while holding data splits and the training budget fixed.

## Code layout

| Path | Contents |
| --- | --- |
| `maize_detector/model.py` | YOLO construction, RPG and neck connections |
| `maize_detector/conditioning.py` | Row descriptors, gains and residual weights |
| `maize_detector/loss.py` | YOLO loss with explicit supervision masks |
| `maize_detector/data.py` | COCO loading and transforms |
| `maize_detector/training.py` | Training, EMA, validation and resume |
| `maize_detector/evaluation.py` | AP and fixed-threshold P/R/F1 |
| `maize_detector/runtime.py` | Checkpoint loading and inference |
| `configs/`, `scripts/`, `tests/` | Configurations, setup utilities and tests |

## Troubleshooting

- **CARAFE or `mmcv._ext` import fails.** Install CUDA-enabled MMCV matching PyTorch 2.1 and CUDA 11.8. Remove `mmcv-lite` if both variants are installed.
- **FreqFusion source is missing.** Run `python scripts/fetch_freqfusion.py`. For another installation location, set `FREQFUSION_SOURCE` to the verified source file.
- **Checkpoint keys differ.** Use the matching scale and module configuration. The final RCFF-YOLO-n/RCFF-YOLO-s tensor key layout is preserved. Other historical architectures need their original implementations.
- **GPU memory is insufficient.** Run one experiment per GPU and keep the physical batch at one. Changing resolution or accumulation changes the supplied experiment configuration.

## Acknowledgments

This implementation builds on [Ultralytics](https://github.com/ultralytics/ultralytics), [FreqFusion](https://github.com/Linwei-Chen/FreqFusion), [MMCV](https://github.com/open-mmlab/mmcv) and [COCO API](https://github.com/cocodataset/cocoapi). See [THIRD_PARTY.md](THIRD_PARTY.md) for attribution and license information.
