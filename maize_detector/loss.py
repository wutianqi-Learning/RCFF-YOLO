"""YOLO task-aligned loss with positive and ignore-region supervision.

The loss preparation follows Ultralytics 8.3.160 v8DetectionLoss (AGPL-3.0).
"""
import torch
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors
from .supervision import ImageSupervision, supervision_weights

class SupervisedYoloLoss(v8DetectionLoss):
    """Native TAL/decode/BboxLoss; adapt only reduction of loss numerators.

    The short preparation mirrors pinned v8DetectionLoss.__call__; native
    preprocessing, TAL, CIoU, DFL and global normalization remain unchanged.
    """
    def __call__(self, preds, batch):
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_dist, scores = torch.cat([x.view(feats[0].shape[0], self.no, -1)
                                     for x in feats], 2).split((self.reg_max * 4, self.nc), 1)
        scores = scores.permute(0, 2, 1).contiguous()
        pred_dist = pred_dist.permute(0, 2, 1).contiguous()
        batch_size = scores.shape[0]
        if len(batch['supervision']) != batch_size or len(batch['ignored_xyxy']) != batch_size:
            raise ValueError('One explicit supervision/ignore record per image required')
        size = torch.tensor(feats[0].shape[2:], device=self.device, dtype=scores.dtype) * self.stride[0]
        anchors, strides = make_anchors(feats, self.stride, .5)
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=size[[1, 0, 1, 0]])
        labels, gt_boxes = targets.split((1, 4), 2)
        mask_gt = gt_boxes.sum(2, keepdim=True).gt_(0.)
        boxes = self.bbox_decode(anchors, pred_dist)
        _, target_boxes, target_scores, fg, _ = self.assigner(
            scores.detach().sigmoid(), (boxes.detach() * strides).type(gt_boxes.dtype),
            anchors * strides, labels, gt_boxes, mask_gt)
        fg = fg.bool()  # Native empty-GT path returns a floating zero mask.
        denominator = max(target_scores.sum(), 1)
        cls_terms = self.bce(scores, target_scores.to(scores.dtype))
        zero = pred_dist.sum() * 0.  # Empty P retains box/DFL graph connections.
        box_loss, cls_loss, dfl_loss = zero, scores.sum() * 0., zero
        for i, record in enumerate(batch['supervision']):
            policy = ImageSupervision.from_coco_image(record)
            weights = supervision_weights(policy, boxes[i].detach() * strides, fg[i],
                batch['ignored_xyxy'][i], tuple(int(x) for x in size.tolist()))
            cls_loss = cls_loss + (cls_terms[i] * weights.classification[:, None]).sum() / denominator
            if fg[i].any():
                native_box, native_dfl = self.bbox_loss(
                    pred_dist[i:i+1], boxes[i:i+1], anchors, target_boxes[i:i+1] / strides,
                    target_scores[i:i+1], denominator, fg[i:i+1])
                box_loss = box_loss + native_box * policy.loss_weight
                dfl_loss = dfl_loss + native_dfl * policy.loss_weight
        losses = torch.stack((box_loss * self.hyp.box, cls_loss * self.hyp.cls,
                              dfl_loss * self.hyp.dfl))
        return losses * batch_size, losses.detach()
