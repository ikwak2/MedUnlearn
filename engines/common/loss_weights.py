"""Training-only class weights; never inspect validation/test frequencies."""
from collections import Counter


def class_weight_spec(rows, classes, mode):
    if mode not in ('none', 'inverse_frequency'):
        raise ValueError('Unknown class weighting')
    counts = Counter(int(r['class_index']) for r in rows)
    if set(counts) != set(range(len(classes))):
        raise ValueError('Every training class must have positive support')
    if any(classes[int(r['class_index'])] != r['label'] for r in rows):
        raise ValueError('Training label/index mismatch')
    n, k = len(rows), len(classes)
    return dict(mode=mode, classes=classes, train_counts=[counts[i] for i in range(k)],
                weights=None if mode == 'none' else [n / (k * counts[i]) for i in range(k)],
                formula='none' if mode == 'none' else 'N_train / (K * n_train_class)',
                reduction='PyTorch CrossEntropyLoss mean; weighted sum divided by sum of target weights per batch',
                frequencies_source='train manifest only')
