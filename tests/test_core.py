import copy
import unittest
import numpy as np
import torch
from maize_detector.conditioning import RowConditioner
from maize_detector.geometry import LetterboxGeometry
from maize_detector.evaluation import evaluate_coco_partition, evaluate_working_point
from maize_detector.optimization import accumulation_divisor, learning_rate, selection_key


class ConditioningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_initial_head_identity_and_gradient(self):
        model = RowConditioner(17)
        inputs = [torch.randn(2, 8, h, h, requires_grad=True) for h in (16, 8, 4)]
        outputs = model(inputs)
        for x, y in zip(inputs, outputs):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        sum(y.square().mean() for y in outputs).backward()
        self.assertGreater(float(model.conditioner[-1].weight.grad.abs().sum()), 0)

    def test_residual_initial_weight_and_independent_parameters(self):
        head, neck = RowConditioner(17), RowConditioner(17)
        self.assertTrue(set(map(id, head.parameters())).isdisjoint(map(id, neck.parameters())))
        high, low = torch.randn(1, 8, 8, 8), torch.randn(1, 8, 4, 4)
        base = torch.nn.functional.interpolate(low, scale_factor=2, mode='nearest')
        fh, fl = torch.randn_like(high), torch.randn_like(base)
        oh, ol = neck.mix_residual(high, base, fh, fl, original_low=low)
        torch.testing.assert_close(oh, high + .1*(fh-high))
        torch.testing.assert_close(ol, base + .1*(fl-base))

    def test_descriptor_boundaries_and_fp32(self):
        layer = RowConditioner().to(dtype=torch.bfloat16)
        self.assertEqual(next(layer.parameters()).dtype, torch.float32)
        descriptor = layer.descriptor(torch.zeros(1, 2, 1, 3), .5)
        self.assertEqual(tuple(descriptor.shape), (1, 4, 1, 1))
        self.assertTrue(torch.isfinite(descriptor).all())
        self.assertEqual(float(descriptor[0, 1, 0, 0]), 0.)


class ProtocolTests(unittest.TestCase):
    def test_schedule_endpoints_and_accumulation_tail(self):
        self.assertAlmostEqual(learning_rate(1), 3e-7)
        self.assertAlmostEqual(learning_rate(1257), 3e-4)
        self.assertAlmostEqual(learning_rate(41900), 3e-6)
        self.assertEqual(accumulation_divisor(3348, 3349), 5)
        self.assertEqual(accumulation_divisor(0, 3349), 8)
        self.assertGreater(selection_key(dict(ap=.4, ap50=.5, f1=.6)),
                           selection_key(dict(ap=.39, ap50=.9, f1=.9)))

    def test_letterbox_roundtrip(self):
        geometry = LetterboxGeometry.from_source(1279, 721, 1280)
        boxes = np.array([[0, 0, 1279, 721], [100, 80, 240, 199]], dtype=float)
        canvas = geometry.to_canvas_xyxy(boxes)
        restored = canvas.copy()
        restored[:, 0::2] = (restored[:, 0::2] - geometry.pad_left) / geometry.scale_x
        restored[:, 1::2] = (restored[:, 1::2] - geometry.pad_top) / geometry.scale_y
        np.testing.assert_allclose(restored, boxes)

    def test_ignore_region_and_empty_predictions(self):
        coco = dict(images=[dict(id=1, width=100, height=100, file_name='test.png', group_id='v1')],
                    categories=[dict(id=1, name='plant')], annotations=[
                        dict(id=1, image_id=1, category_id=1, bbox=[10, 10, 20, 20], area=400, iscrowd=0),
                        dict(id=2, image_id=1, category_id=1, bbox=[60, 60, 30, 30], area=900, iscrowd=1)])
        predictions = [dict(image_id=1, category_id=1, bbox=box, score=.9)
                       for box in ([10, 10, 20, 20], [65, 65, 10, 10])]
        scores = evaluate_working_point(coco, predictions, threshold=.25)
        self.assertEqual((scores['tp'], scores['fp'], scores['fn']), (1, 0, 0))
        ap = evaluate_coco_partition(coco, predictions)
        self.assertAlmostEqual(ap['ap50'], 1.)
        empty = evaluate_working_point(coco, [], threshold=.25)
        self.assertEqual((empty['tp'], empty['fp'], empty['fn']), (0, 0, 1))
        self.assertEqual(evaluate_coco_partition(coco, [])['ap'], 0.)


if __name__ == '__main__':
    unittest.main()
