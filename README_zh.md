# RCFF-YOLO

[English](README.md)

RCFF-YOLO 为逐行条件频率融合模型（Row-Conditioned Frequency Fusion YOLO）。对应论文为 *Mitigating Perspective Scale Attenuation and Canopy Aliasing: Row-Conditioned Frequency Fusion for Maize Seedling Detection*。

本仓库提供用于农机前视玉米幼苗检测的 **RCFF-YOLO-n** 和 **RCFF-YOLO-s**。模型以 YOLO11 为基础，采用 RPG 调节进入检测头的多尺度特征，通过 FreqFusion 增强颈部特征。两个条件网络具有独立参数，分别控制特征增益与频率特征的残差混合。最终模型保留可学习坐标重采样。

仓库包含模型、训练、评估、单图推理、配置文件和测试。数据集与训练权重需要单独准备。

## 安装

参考环境为 Linux、Python 3.11、PyTorch 2.1.0、CUDA 11.8、Ultralytics 8.3.160 和 MMCV 2.1.0。FreqFusion 需要带 CUDA CARAFE 算子的 MMCV。

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

下载脚本获取固定版本的官方 FreqFusion 源码，并校验文件哈希。来源与许可证说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。

```bash
python -m unittest discover -s tests -v
python scripts/smoke_model.py --scale n --device cuda:0
```

## 数据与训练

使用单类别 COCO 格式，类别名称为 `plant`，类别编号为 1。图像记录视频组、划分和标签来源，具体格式见[数据说明](docs/data.md)。MaizeHorizon 训练采用公开 nearfar 标签，其中包含人工标注与传播标签；验证和测试采用独立视频组中的人工标注。当前发布配置不混合 Maize-FV。

```bash
cp configs/data.example.yaml configs/data.local.yaml
```

在本地配置中填写图像根目录、训练标注和验证标注路径。将官方 COCO 预训练权重放入 `weights/`。RCFF-YOLO-s 使用 `yolo11s.pt`，RCFF-YOLO-n 使用 `yolo11n.pt`。

```bash
python -m maize_detector train \
  --config configs/rcff_yolo_s.yaml --data configs/data.local.yaml \
  --pretrained weights/yolo11s.pt \
  --output outputs/rcff_yolo_s_r1 --device cuda:0
```

训练配置为 100 轮、1280 × 1280 输入、AdamW 优化器、峰值学习率 0.0003 和权重衰减 0.0001。前三轮预热，随后余弦衰减。物理批量为 1，通过 8 次梯度累积形成有效批量 8。损失平衡因子为 box 7.5、cls 0.5 和 dfl 1.5。

权重依据验证集 mAP50:95 选择，数值相同时依次比较 mAP50 和 F1。`best.pt` 保存 EMA 权重，`last.pt` 用于轮次边界恢复，`history.jsonl` 保存每轮损失与验证指标。

三次独立训练采用 `110022893`、`1970871798` 和 `1114612310`。通过 `--seed` 指定，每次均加载相同的公开 COCO 权重。各运行使用独立输出目录，可分配至不同 GPU。

```bash
python -m maize_detector train \
  --config configs/rcff_yolo_s.yaml --data configs/data.local.yaml \
  --resume outputs/rcff_yolo_s_r1/last.pt \
  --output outputs/rcff_yolo_s_r1 --device cuda:0
```

恢复时保持配置、数据路径、随机初始化状态和输出目录一致。

## 评估与推理

```bash
python -m maize_detector evaluate \
  --config configs/rcff_yolo_s.yaml --weights outputs/rcff_yolo_s_r1/best.pt \
  --annotations data/maizehorizon/annotations/test.json \
  --image-root data/maizehorizon/images \
  --output outputs/rcff_yolo_s_r1/test --device cuda:0
```

输出包括指标与预测 JSON。AP 按 COCO 协议计算。P、R 采用置信度 0.25 和 IoU 0.50，NMS 的 IoU 阈值为 0.70。JSON 指标为小数，转为百分数时乘以 100。三次训练分别评估，再计算均值和样本标准差。

```bash
python -m maize_detector predict \
  --config configs/rcff_yolo_s.yaml --weights outputs/rcff_yolo_s_r1/best.pt \
  --image data/example.jpg --output outputs/example --device cuda:0
```

单图推理保存预测框可视化与预测 JSON。`--confidence` 控制显示阈值，默认 0.25。

## 配置与代码

`configs/` 包含 n、s 两种规模的基线、单独 RPG、条件残差 FreqFusion 和完整融合配置，均采用所附的 100 轮训练配方。最终网络见 [model.py](maize_detector/model.py)，行描述子与条件网络见 [conditioning.py](maize_detector/conditioning.py)。完整目录说明和常见问题见[英文 README](README.md)。

若提示 CARAFE 或 `mmcv._ext` 缺失，应安装与 PyTorch、CUDA 对应的完整 MMCV，不能以 `mmcv-lite` 替代。当前仓库不提供训练权重下载地址。
