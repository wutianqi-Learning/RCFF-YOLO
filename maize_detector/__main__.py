"""Command-line entry point for training, evaluation and image inference."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog='python -m maize_detector')
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('train', 'evaluate', 'predict'):
        sub = commands.add_parser(command)
        sub.add_argument('--config', required=True)
        sub.add_argument('--device', default='cuda:0')
        sub.add_argument('--output', required=True)
        if command == 'train':
            sub.add_argument('--data', required=True)
            sub.add_argument('--pretrained')
            sub.add_argument('--resume')
            sub.add_argument('--seed', type=int)
        else:
            sub.add_argument('--weights', required=True)
            sub.add_argument('--precision', choices=('bf16', 'fp32'), default='bf16')
        if command == 'evaluate':
            sub.add_argument('--annotations', required=True)
            sub.add_argument('--image-root', required=True)
        elif command == 'predict':
            sub.add_argument('--image', required=True)
            sub.add_argument('--confidence', type=float, default=.25)
    args = parser.parse_args()
    from .runtime import read_config, read_json, load_model, evaluate, predict_coco
    config = read_config(args.config)
    if args.command == 'train':
        from .training import train
        if args.seed is not None:
            config['seed'] = args.seed
        train(config, read_config(args.data), args.output, args.pretrained, args.device, args.resume)
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = load_model(config, args.weights, args.device)
    if args.command == 'evaluate':
        coco = read_json(args.annotations)
        metrics, predictions = evaluate(model, coco, args.image_root, args.device,
                                       config['image_size'], args.precision)
        (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        (output / 'predictions.json').write_text(json.dumps(predictions, indent=2) + '\n')
        print(json.dumps(metrics, indent=2))
    else:
        from PIL import Image, ImageDraw
        image_path = Path(args.image)
        with Image.open(image_path) as original:
            image = original.convert('RGB')
        coco = dict(images=[dict(id=1, file_name=image_path.name,
                                width=image.width, height=image.height)])
        predictions = predict_coco(model, coco, image_path.parent, args.device,
                                   config['image_size'], args.precision)
        draw = ImageDraw.Draw(image)
        for row in predictions:
            if row['score'] >= args.confidence:
                x, y, w, h = row['bbox']
                draw.rectangle((x, y, x+w, y+h), outline='#2a9d8f', width=3)
                draw.text((x, max(0, y-12)), f"plant {row['score']:.2f}", fill='#2a9d8f')
        image.save(output / 'prediction.png')
        (output / 'predictions.json').write_text(json.dumps(predictions, indent=2) + '\n')


if __name__ == '__main__':
    main()
