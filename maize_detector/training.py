"""Training with an update-wise cosine schedule and EMA model selection."""
import json
import math
from pathlib import Path
import random
import numpy as np
import torch
from .data import CocoTrainingDataset, collate_training
from .optimization import (EMAState, accumulation_divisor, apply_ema, atomic_json,
                           learning_rate, parameter_groups, selection_key)
from .runtime import build_model, evaluate, precision_context, read_json


def seed_training(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else
            [item.to(device) for item in value] if key == 'ignored_xyxy' else value
            for key, value in batch.items()}


def _validate_splits(train_coco, validation):
    train = read_json(train_coco)
    if any(row.get('split') != 'validation' or row.get('annotation_origin') != 'human'
           for row in validation['images']):
        raise ValueError('Weight selection requires manually annotated validation images')
    train_groups = {row.get('group_id') for row in train['images']}
    val_groups = {row.get('group_id') for row in validation['images']}
    if None in train_groups | val_groups or train_groups & val_groups:
        raise ValueError('Train and validation require disjoint video group IDs')
    train_files = {row['file_name'] for row in train['images']}
    if train_files & {row['file_name'] for row in validation['images']}:
        raise ValueError('Train and validation contain the same image names')


def train(config, data, output, pretrained, device='cuda:0', resume=None):
    recipe = config['training']
    seed, epochs = int(config['seed']), int(recipe['epochs'])
    accumulation = int(recipe['accumulation'])
    if recipe['batch_size'] != 1 or accumulation < 1 or epochs < 1:
        raise ValueError('The supplied recipe uses microbatch size 1 and positive accumulation')
    device = torch.device(device)
    torch.set_num_threads(recipe.get('cpu_threads', 4))
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    seed_training(seed)
    validation = read_json(data['validation'])
    _validate_splits(data['train'], validation)
    dataset = CocoTrainingDataset(data['train'], data['image_root'],
                                  config['image_size'], recipe['horizontal_flip'])
    batches = len(dataset)
    updates_per_epoch = math.ceil(batches / accumulation)
    total_updates = epochs * updates_per_epoch
    warmup_updates = int(recipe['warmup_epochs']) * updates_per_epoch
    if not 1 <= warmup_updates < total_updates:
        raise ValueError('Warmup must occupy at least one update and fewer than all epochs')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not resume and any(output.iterdir()):
        raise FileExistsError('Use a new output directory or --resume for an existing run')
    model = build_model(config)
    if not resume:
        if not pretrained:
            raise ValueError('--pretrained must point to official COCO weights')
        model.load_official_coco_tuning(pretrained)
    model = model.to(device).train()
    groups, _ = parameter_groups(model, recipe['weight_decay'])
    optimizer = torch.optim.AdamW(groups, lr=recipe['learning_rate'], betas=(.9, .999), eps=1e-8)
    parameters = [p for p in model.parameters() if p.requires_grad]
    ema = EMAState(model)
    epoch_start, update, best_key, best_epoch = 1, 0, None, None
    seed_training(seed)
    if resume:
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if saved['config'] != config or saved['data'] != data:
            raise ValueError('Resume configuration or dataset paths changed')
        model.load_state_dict(saved['raw_model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        ema.shadow = {k: v.to(device) for k, v in saved['ema']['shadow'].items()}
        ema.updates = saved['ema']['updates']
        epoch_start, update = saved['epoch'] + 1, saved['update']
        best_key, best_epoch = saved['best_key'], saved['best_epoch']
        torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
    atomic_json(output / 'config.json', dict(config=config, data=data))
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epoch_start, epochs + 1):
        generator = torch.Generator().manual_seed((seed + epoch * 104729) % (2**63 - 1))
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True,
            generator=generator, num_workers=recipe['workers'], pin_memory=device.type == 'cuda',
            collate_fn=collate_training, persistent_workers=False)
        sums = dict(loss_box=0., loss_cls=0., loss_dfl=0.)
        model.train()
        for index, packed in enumerate(loader):
            if index % accumulation == 0:
                rate = learning_rate(update + 1, total_updates=total_updates,
                    warmup_updates=warmup_updates, initial=recipe['learning_rate'],
                    final_factor=recipe['final_lr_factor'])
                for group in optimizer.param_groups:
                    group['lr'] = rate
            batch = device_batch(packed['inputs'], device)
            with precision_context(device, recipe['precision']):
                components = model(batch, mode='loss')
                loss = sum(components.values())
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            for key, value in components.items():
                sums[key] += float(value.detach())
            (loss / accumulation_divisor(index, len(loader), accumulation)).backward()
            if (index + 1) % accumulation and index + 1 != len(loader):
                continue
            norm = torch.nn.utils.clip_grad_norm_(parameters, recipe['clip_norm'])
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite training gradients')
            optimizer.step()
            update += 1
            ema.update(model, update)
            optimizer.zero_grad(set_to_none=True)
        with apply_ema(model, ema):
            model.eval()
            metrics, _ = evaluate(model, validation, data['image_root'], device,
                                  config['image_size'], recipe['precision'])
            key = selection_key(metrics)
            if best_key is None or key > tuple(best_key):
                best_key, best_epoch = key, epoch
                torch.save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                                config=config, epoch=epoch, validation=metrics), output / 'best.pt')
        record = dict(epoch=epoch, update=update, learning_rate=rate, validation=metrics,
                      loss={k: v / len(loader) for k, v in sums.items()}, best_epoch=best_epoch)
        with (output / 'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        atomic_json(output / 'progress.json', record)
        saved = dict(config=config, data=data, epoch=epoch, update=update,
            raw_model={k: v.detach().cpu() for k, v in model.state_dict().items()},
            optimizer=optimizer.state_dict(), ema=ema.state_dict(cpu=True),
            best_key=best_key, best_epoch=best_epoch, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
            numpy_rng=np.random.get_state(), python_rng=random.getstate())
        temporary = output / 'last.partial.pt'
        torch.save(saved, temporary)
        temporary.replace(output / 'last.pt')
        print(json.dumps(record, allow_nan=False), flush=True)
    return best_epoch
