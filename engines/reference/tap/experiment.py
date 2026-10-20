"""Frozen HYPO P1/P5 and AR references, separate from the common benchmark."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import resnet18

from classwise_metrics import compare, evaluate, read_predictions, write_csv

CODE = Path(__file__).resolve().parent
WORK = Path(os.environ.get('HYPO_WORK', '/work'))
INPUTS = Path(os.environ.get('HYPO_INPUTS', '/inputs'))
UPSTREAM = '8f033e000d85f93c34ed4a55f020952b240da235'
PROTOCOL = 'hypo_official_linf_reference_v1'
AR_UPSTREAM = 'bf350b0cf2dc8ef4786a38950b994320dc450da6'
RECIPE = 'HYPO'
TASKS = {'cifar10': 'CIFAR-10', 'mri4': 'MRI4', 'mri15': 'MRI15', 'xrayfixed': 'X-ray'}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


steps = module('hypo_official_steps', CODE/'official/attacks/step.py')
official_resnet = module('hypo_official_resnet', CODE/'official/models/resnet.py')
ar_resnet = module('ar_official_resnet', CODE/'ar_official/models/resnet.py')
official_vgg = module('hypo_official_vgg', CODE/'official/models/vgg.py')


def protocol():
    return 'ar_official_l2_reference_v1' if RECIPE == 'AR' else PROTOCOL


def reference_gate(method, clean, poison):
    return clean >= .90 and poison <= {'P1': .25, 'P5': .20, 'AR': .25}[method]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_num_threads(4)


def model_for(task, classes):
    if task == 'cifar10':
        source = ar_resnet if RECIPE == 'AR' else official_resnet
        return source.ResNet18() if RECIPE == 'AR' else source.ResNet18(num_classes=len(classes))
    # Explicit medical adaptation: keep the existing 256px medical architecture.
    return resnet18(weights=None, num_classes=len(classes))


def surrogate_for(task, classes):
    model = official_vgg.VGG16(num_classes=len(classes))
    if task != 'cifar10':
        model.features[-1] = torch.nn.AdaptiveAvgPool2d(1)
    return model


def p1(model, x, y, classes, epsilon=8/255, iterations=100):
    original = x.detach().clone()
    step = steps.LinfStep(original, epsilon, epsilon/5)
    targets = (y+1) % classes
    for _ in range(iterations):
        x = x.detach().requires_grad_(True)
        loss = F.cross_entropy(model(x), targets)
        gradient = torch.autograd.grad(loss, x)[0]
        with torch.no_grad():
            x = step.project(step.step(x, gradient)).clamp(0, 1)
    return x.detach()


def p5(classes, shape, device='cpu', epsilon=8/255):
    zeros = torch.zeros(classes, *shape, device=device)
    return steps.LinfStep(None, epsilon, None).random_perturb(zeros)


class Images(Dataset):
    def __init__(self, images, labels, indices=None, augment=False, normalization=None):
        self.images, self.labels = images, labels
        self.indices = list(range(len(labels))) if indices is None else indices
        self.transform = transforms.Compose(([
            transforms.RandomCrop(images.shape[-1], padding=4),
            transforms.RandomHorizontalFlip(),
        ] if augment else []) + [transforms.ToTensor()] +
            ([transforms.Normalize(*normalization)] if normalization is not None else []))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        i = self.indices[j]
        array = np.asarray(self.images[i])
        # Official CIFAR10Poisoned quantizes float poisons through uint8/PIL.
        if array.dtype != np.uint8:
            array = (array*255).astype(np.uint8)
        image = Image.fromarray(array.transpose(1, 2, 0))
        return self.transform(image), int(self.labels[i]), int(i)


def load_data(task):
    if task == 'cifar10':
        train = datasets.CIFAR10(os.environ['MEDUNLEARN_CIFAR_ROOT'], train=True, download=False)
        test = datasets.CIFAR10(os.environ['MEDUNLEARN_CIFAR_ROOT'], train=False, download=False)
        all_indices = torch.randperm(50000, generator=torch.Generator().manual_seed(0)).tolist()
        return dict(classes=train.classes, train_x=train.data.transpose(0, 3, 1, 2),
                    train_y=np.array(train.targets), test_x=test.data.transpose(0, 3, 1, 2),
                    test_y=np.array(test.targets),
                    train_indices=list(range(50000)) if RECIPE=='AR' else all_indices[:49000],
                    val_indices=None if RECIPE=='AR' else all_indices[49000:], val_x=None, val_y=None,
                    test_ids=[f'cifar10:test:{i}' for i in range(10000)],
                    normalization=((.4914,.4822,.4465),(.2023,.1994,.2010)) if RECIPE=='AR' else None)
    root = INPUTS/task
    dataset = TASKS[task]
    classes = json.loads((root/'splits_v1'/dataset/'classes.json').read_text())
    arrays, labels, manifests = {}, {}, {}
    for split in ['train', 'validation', 'test']:
        path = root/'splits_v1'/dataset/(split+'.csv')
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        cache = root/'cache'/dataset/(split+'.npy')
        meta = json.loads(cache.with_suffix('.json').read_text())
        assert meta['split_sha256'] == digest(path), 'Stale medical cache'
        arrays[split] = np.load(cache, mmap_mode='r')
        assert arrays[split].dtype == np.uint8
        assert arrays[split].shape == (len(rows), 3, 256, 256)
        labels[split] = np.array([int(row['class_index']) for row in rows])
        assert set(labels[split]) == set(range(len(classes)))
        manifests[split] = {row['sample_id'] for row in rows}
        assert len(manifests[split]) == len(rows)
        if split == 'test':
            test_ids = [row['sample_id'] for row in rows]
    assert all(not manifests[a] & manifests[b] for a, b in
               [('train', 'validation'), ('train', 'test'), ('validation', 'test')])
    if task == 'xrayfixed':
        assert len(classes) == 12
        assert [len(arrays[s]) for s in ['train', 'validation', 'test']] == [8843, 3758, 8417]
    normalization = None
    if RECIPE == 'AR':
        count, total, squared = 0, np.zeros(3), np.zeros(3)
        for begin in range(0, len(arrays['train']), 32):
            pixels = np.array(arrays['train'][begin:begin+32], dtype=np.float32)/255
            total += pixels.sum(axis=(0,2,3), dtype=np.float64)
            squared += np.square(pixels).sum(axis=(0,2,3), dtype=np.float64)
            count += pixels.shape[0]*pixels.shape[2]*pixels.shape[3]
        mean = total/count
        std = np.sqrt(np.maximum(squared/count-mean*mean, 1e-12))
        normalization = (mean.tolist(), std.tolist())
    return dict(classes=classes, train_x=arrays['train'], train_y=labels['train'],
                test_x=arrays['test'], test_y=labels['test'],
                train_indices=list(range(len(labels['train']))), val_indices=None,
                val_x=arrays['validation'], val_y=labels['validation'], test_ids=test_ids,
                normalization=normalization)


def loader(data, batch, shuffle=False, generator=None):
    return DataLoader(data, batch_size=batch, shuffle=shuffle, num_workers=4,
                      pin_memory=True, generator=generator)


def predict(model, data, classes, batch, ids=None):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for x, y, indices in loader(data, batch):
            logits = model(x.cuda(non_blocking=True))
            assert torch.isfinite(logits).all(), 'Nonfinite evaluation'
            for i, actual, pred in zip(indices.tolist(), y.tolist(), logits.argmax(1).cpu().tolist()):
                predictions.append(dict(sample_id=str(i) if ids is None else ids[i],
                                        y_true=classes[actual], y_pred=classes[pred]))
    return predictions, evaluate(predictions, classes)


def checkpoint(path, value):
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(path)


def train(task, name, data, train_x, root):
    seed_all()
    out = root/name
    out.mkdir(parents=True, exist_ok=True)
    reference = task == 'cifar10'
    surrogate = name == 'SURROGATE'
    ar = RECIPE == 'AR'
    epochs = 100 if ar else 150
    batch = 128 if reference else 32
    config = dict(protocol=protocol(), upstream_commit=AR_UPSTREAM if ar else UPSTREAM,
                  task=task, condition=name,
                  seed=0, epochs=epochs, optimizer='SGD', lr=.1, momentum=.9,
                  weight_decay=.0005, milestones=[50,100] if ar else [100,125], gamma=.1,
                  batch_size=batch, class_weighting='none',
                  augmentation='PIL RandomCrop(size,padding=4),RandomHorizontalFlip,ToTensor',
                  normalization=data['normalization'], quantization='HYPO truncates; AR rounds as official PNG export',
                  model='official VGG16; medical adaptive final pooling' if surrogate else
                        'official ResNet18' if reference else 'torchvision medical ResNet18',
                  selection='fixed final; supplementary clean validation BA on medical only' if ar else
                            'max augmented holdout accuracy; earliest tie' if reference else
                            'max clean validation balanced accuracy; earliest tie',
                  reference_validation='no holdout; all 50000; no test monitoring' if reference and ar else
                                       '1000 held-out images; UE holdout is poisoned and augmented' if reference else
                                       'existing clean disjoint validation; train-only generation',
                  test_use='after training only; validation-best and fixed-final reported separately',
                  torch=torch.__version__, cuda=torch.version.cuda, host=os.environ.get('MEDUNLEARN_HOST','local'),
                  gpu_uuid=os.environ.get('ASSIGNED_GPU_UUID'))
    if not reference:
        config['split_sha256'] = {s: digest(INPUTS/task/'splits_v1'/TASKS[task]/(s+'.csv'))
                                  for s in ['train', 'validation', 'test']}
    if (out/'config.json').exists():
        assert json.loads((out/'config.json').read_text()) == config, 'Changed training config'
    atomic_json(out/'config.json', config)
    if (out/'complete.json').exists():
        return out
    train_ds = Images(train_x, data['train_y'], data['train_indices'], augment=True,
                      normalization=data['normalization'])
    val_ds = (None if reference and ar else
              Images(train_x, data['train_y'], data['val_indices'], augment=True,
                     normalization=data['normalization']) if reference else
              Images(data['val_x'], data['val_y'], normalization=data['normalization']))
    generator = torch.Generator().manual_seed(0)
    train_loader = loader(train_ds, batch, True, generator)
    model = (surrogate_for(task, data['classes']) if surrogate else model_for(task, data['classes'])).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100] if ar else [100,125], gamma=.1)
    start, best, history = 0, -1., []
    if (out/'last.pt').exists():
        saved = torch.load(out/'last.pt', map_location='cpu')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        start, best, history = saved['epoch']+1, saved['best'], saved['history']
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        generator.set_state(saved['loader_rng'])
    for epoch in range(start, epochs):
        begun = time.monotonic()
        model.train()
        total, correct, loss_sum = 0, 0, 0.
        lr = optimizer.param_groups[0]['lr']
        for x, y, _ in train_loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            assert torch.isfinite(loss), 'Nonfinite training loss'
            loss.backward()
            optimizer.step()
            total += len(y)
            correct += (logits.argmax(1) == y).sum().item()
            loss_sum += loss.item()*len(y)
        metrics = predict(model, val_ds, data['classes'], batch)[1] if val_ds is not None else {}
        score = metrics.get('accuracy' if reference else 'balanced_accuracy', -1.)
        row = dict(epoch=epoch+1, train_loss=loss_sum/total, train_accuracy=correct/total,
                   validation_accuracy=metrics.get('accuracy'),
                   validation_balanced_accuracy=metrics.get('balanced_accuracy'), lr=lr,
                   seconds=time.monotonic()-begun)
        history.append(row)
        if val_ds is not None and score > best:
            best = score
            checkpoint(out/'best.pt', dict(model=model.state_dict(), epoch=epoch, config=config))
        scheduler.step()
        checkpoint(out/'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                   scheduler=scheduler.state_dict(), epoch=epoch, best=best, history=history,
                   python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                   torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                   loader_rng=generator.get_state()))
        atomic_json(out/'history.json', history)
        atomic_json(out/'status.json', dict(row, state='training', epochs=epochs))
        print(task, name, json.dumps(row), flush=True)
    if surrogate:
        saved = torch.load(out/'best.pt', map_location='cpu')
        e = dict(protocol=protocol(), selected_epoch=saved['epoch']+1, completed_at=time.time())
        atomic_json(out/'complete.json', e)
        atomic_json(out/'status.json', dict(e, state='completed'))
        return out
    test = Images(data['test_x'], data['test_y'], normalization=data['normalization'])
    report = {}
    selections = ([('validation_best', out/'best.pt')] if val_ds is not None else []) + [('fixed_final', out/'last.pt')]
    for selection, path in selections:
        saved = torch.load(path, map_location='cpu')
        model.load_state_dict(saved['model'])
        predictions, metrics = predict(model, test, data['classes'], batch, data['test_ids'])
        if name != 'CLEAN':
            clean = read_predictions(root/'CLEAN'/f'{selection}_predictions.csv')
            _, metrics = compare(clean, predictions, data['classes'])
        write_csv(out/f'{selection}_predictions.csv', predictions)
        report[selection] = dict(metrics=metrics, selected_epoch=saved['epoch']+1)
    atomic_json(out/'metrics.json', report)
    atomic_json(out/'complete.json', dict(completed_at=time.time(), protocol=protocol(), reports=report))
    atomic_json(out/'status.json', dict(state='completed', reports=report))
    return out


def generate(task, method, data, root):
    seed_all()
    out = root/'poisons'
    out.mkdir(exist_ok=True)
    destination = out/(method+'.npy')
    if destination.exists() and destination.with_suffix('.json').exists():
        meta = json.loads(destination.with_suffix('.json').read_text())
        assert meta['sha256'] == digest(destination), 'Poison hash mismatch'
        return np.load(destination, mmap_mode='r')
    x = data['train_x']
    shape = x.shape
    metadata = dict(method=method, protocol=PROTOCOL, epsilon=8/255, norm='Linf', seed=0,
                    iterations=100 if method=='P1' else 1, step_size=(8/255)/5 if method=='P1' else None,
                    target='(y+1)%class_count' if method=='P1' else None,
                    distribution='class-specific Uniform[-8/255,8/255]' if method=='P5' else None,
                    generation_samples=len(x), test_used=False,
                    scope='all 50000 including reference holdout' if task=='cifar10' else 'corrected train only')
    model = None
    if method == 'P1':
        model = surrogate_for(task, data['classes']).cuda()
        saved = torch.load(root/'SURROGATE/best.pt', map_location='cpu')
        model.load_state_dict(saved['model'])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        metadata['surrogate_sha256'] = digest(root/'SURROGATE/best.pt')
        metadata['surrogate_model'] = 'official VGG16; clean validation-selected'
    else:
        universal = p5(len(data['classes']), shape[1:], device='cuda').cpu()
        np.save(out/'P5_universal.npy', universal.numpy())
    partial = destination.with_suffix('.partial.npy')
    values = np.lib.format.open_memmap(partial, mode='w+', dtype='float32', shape=shape)
    batch = 256 if task=='cifar10' else 32
    maximum = quantized_maximum = 0.
    for begin in range(0, len(x), batch):
        pixels = torch.from_numpy(np.array(x[begin:begin+batch], copy=True)).float()/255
        labels = torch.tensor(data['train_y'][begin:begin+batch], dtype=torch.long)
        if model is not None:
            result = p1(model, pixels.cuda(), labels.cuda(), len(data['classes'])).cpu()
        else:
            result = (pixels + universal[labels]).clamp(0, 1)
        assert torch.isfinite(result).all()
        observed = (result-pixels).abs().max().item()
        assert observed <= 8/255+1e-6, 'Poison exceeds Linf budget before official quantization'
        maximum = max(maximum, observed)
        quantized = (result*255).to(torch.uint8).float()/255
        quantized_maximum = max(quantized_maximum, (quantized-pixels).abs().max().item())
        values[begin:begin+len(result)] = result.numpy()
        progress = dict(state='generating', task=task, method=method,
                        done=min(begin+batch, len(x)), total=len(x), observed_max_linf=maximum)
        atomic_json(out/'status.json', progress)
        if begin % (batch*10) == 0:
            print(json.dumps(progress), flush=True)
    values.flush()
    del values
    partial.replace(destination)
    metadata.update(observed_max_linf=maximum, observed_quantized_max_linf=quantized_maximum,
                    sha256=digest(destination), upstream_commit=UPSTREAM)
    atomic_json(destination.with_suffix('.json'), metadata)
    return np.load(destination, mmap_mode='r')


def generate_ar(task, data, root):
    from ar_generation import recurrence
    seed_all()
    out = root/'poisons'
    out.mkdir(exist_ok=True)
    destination = out/'AR.npy'
    if destination.exists() and destination.with_suffix('.json').exists():
        meta = json.loads(destination.with_suffix('.json').read_text())
        assert meta['sha256'] == digest(destination), 'AR poison hash mismatch'
        return np.load(destination, mmap_mode='r')
    x = data['train_x']
    coefficients = np.array(json.loads((CODE/'ar_coefficients.json').read_text())[task], dtype=np.float32)
    assert coefficients.shape == (len(data['classes']), 3, 3, 3)
    assert np.all(coefficients[:, :, 2, 2] == 0)
    partial = destination.with_suffix('.partial.npy')
    values = np.lib.format.open_memmap(partial, mode='w+', dtype='uint8', shape=x.shape)
    crop = 4
    maximum = quantized_maximum = 0.
    for begin in range(0, len(x), 32):
        labels = np.asarray(data['train_y'][begin:begin+32])
        initial = torch.randn(len(labels), 3, x.shape[2]+crop, x.shape[3]+crop).numpy()
        signals = recurrence(initial, coefficients[labels])[:, :, crop:, crop:]
        norms = np.linalg.norm(signals.reshape(len(labels), -1), axis=1)
        assert np.isfinite(norms).all() and (norms > 0).all(), 'Unstable AR process'
        delta = signals/norms[:, None, None, None]
        original = np.asarray(x[begin:begin+len(labels)], dtype=np.float32)/255
        poisoned = np.clip(original+delta, 0, 1)
        observed = np.linalg.norm((poisoned-original).reshape(len(labels), -1), axis=1).max()
        assert observed <= 1.+1e-4
        maximum = max(maximum, float(observed))
        quantized = np.clip(poisoned*255+.5, 0, 255).astype(np.uint8)
        effective = np.linalg.norm((quantized.astype(np.float32)/255-original).reshape(len(labels), -1), axis=1).max()
        quantized_maximum = max(quantized_maximum, float(effective))
        values[begin:begin+len(labels)] = quantized
        progress = dict(state='generating', task=task, method='AR',
                        done=min(begin+32, len(x)), total=len(x), observed_max_l2=maximum)
        atomic_json(out/'status.json', progress)
        if begin % 320 == 0:
            print(json.dumps(progress), flush=True)
    values.flush()
    del values
    partial.replace(destination)
    metadata = dict(method='AR', protocol=protocol(), epsilon=1., norm='L2', seed=0,
                    observed_max_l2=maximum, observed_quantized_max_l2=quantized_maximum,
                    sha256=digest(destination), upstream_commit=AR_UPSTREAM,
                    coefficient_json_sha256=digest(CODE/'ar_coefficients.json'),
                    coefficient_source='upstream hardcoded RNMR10' if task=='cifar10' else
                                       'existing medical mr5 coefficients; fixed X-ray retains parent class mapping',
                    numerical_implementation='float32 Numba recurrence; compared with official torch conv2d',
                    crop=crop, quantization='round as official PNG export',
                    generation_samples=len(x), test_used=False,
                    scope='all 50000' if task=='cifar10' else 'corrected train only')
    atomic_json(destination.with_suffix('.json'), metadata)
    return np.load(destination, mmap_mode='r')


def export(task, method, root):
    public = root/'public'
    public.mkdir(exist_ok=True)
    summaries, details = [], []
    for name in ['CLEAN', method]:
        reports = json.loads((root/name/'metrics.json').read_text())
        for selection, result in reports.items():
            metrics = result['metrics']
            summaries.append(dict(task=task, method=name, selection=selection, protocol=protocol(),
                                  accuracy_pct=100*metrics['accuracy'],
                                  balanced_accuracy_pct=100*metrics['balanced_accuracy'],
                                  macro_f1_pct=100*metrics['macro_f1'],
                                  ba_drop_pp=metrics.get('balanced_accuracy_drop_pp', 0),
                                  selected_epoch=result['selected_epoch'], test_n=metrics['n']))
            for i, label in enumerate(metrics['classes']):
                details.append(dict(task=task, method=name, selection=selection, class_label=label,
                                    test_n=metrics['support'][i], recall_pct=100*metrics['recall'][i],
                                    recall_drop_pp=metrics.get('recall_drop_pp', [0]*len(metrics['classes']))[i]))
    write_csv(public/'comparison.csv', summaries)
    write_csv(public/'classwise.csv', details)
    atomic_json(public/'generation.json', json.loads((root/'poisons'/f'{method}.json').read_text()))


def run(task, method):
    global RECIPE
    RECIPE = 'AR' if method=='AR' else 'HYPO'
    assert torch.cuda.device_count() == 1, 'Exactly the assigned GPU must be visible'
    root = WORK/task
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root/'environment.json', dict(torch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0),
                uuid=os.environ.get('ASSIGNED_GPU_UUID')))
    data = load_data(task)
    train(task, 'CLEAN', data, data['train_x'], root)
    if method=='P1':
        train(task, 'SURROGATE', data, data['train_x'], root)
    poison = generate_ar(task, data, root) if method=='AR' else generate(task, method, data, root)
    train(task, method, data, poison, root)
    export(task, method, root)
    marker = dict(state='completed', task=task, method=method, protocol=protocol(), completed_at=time.time())
    if task == 'cifar10':
        selection = 'fixed_final' if method=='AR' else 'validation_best'
        clean = json.loads((root/'CLEAN/metrics.json').read_text())[selection]['metrics']['accuracy']
        ue = json.loads((root/method/'metrics.json').read_text())[selection]['metrics']['accuracy']
        # Predeclared reference sanity gate, never a medical-test selection rule.
        gate = reference_gate(method, clean, ue)
        marker.update(reference_gate_passed=gate, clean_accuracy=clean, poison_accuracy=ue,
                      gate_definition='CLEAN>=90%; P1<=25% / P5<=20% / AR<=25%; fixed before first run')
    atomic_json(root/'complete.json', marker)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=list(TASKS), required=True)
    parser.add_argument('--method', choices=['P1', 'P5', 'AR'], required=True)
    args = parser.parse_args()
    run(args.task, args.method)
