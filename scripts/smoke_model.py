"""Run a small CUDA forward/backward check for Ours-n or Ours-s."""
import argparse
import torch
from maize_detector.model import MaizeDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', choices=('n', 's'), default='n')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = MaizeDetector(args.scale).to(args.device).train()
    inputs = dict(img=torch.rand(1, 3, 128, 128, device=args.device),
        bboxes=torch.tensor([[.5, .5, .3, .3]], device=args.device),
        cls=torch.zeros(1, 1, device=args.device), batch_idx=torch.zeros(1, device=args.device),
        supervision=[dict(maize_annotation_origin='human', maize_loss_weight=1.,
                          maize_supervision_policy='human_positive_and_background_except_ignore')],
        ignored_xyxy=[torch.empty(0, 4, device=args.device)])
    with torch.autocast('cuda', dtype=torch.bfloat16):
        losses = model(inputs)
    sum(losses.values()).backward()
    for name in ('perspective_gate', 'frequency_fusion', 'row_neck_conditioner'):
        gradients = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError(f'Invalid gradients in {name}')
        if sum(float(g.abs().sum()) for g in gradients) == 0:
            raise RuntimeError(f'No gradient signal in {name}')
    print({key: float(value.detach()) for key, value in losses.items()})
    print('Forward/backward check passed')


if __name__ == '__main__':
    main()
