"""Apply the unchanged official AA loss in pixel space, then native input normalization."""
import torch
from aa_loss import training_loss, AA_CONFIG


class NormalizedModel(torch.nn.Module):
    def __init__(self, model, stats):
        super().__init__(); self.model = model; self.stats = stats

    def forward(self, pixels):
        mean = pixels.new_tensor(self.stats[0])[None, :, None, None]
        std = pixels.new_tensor(self.stats[1])[None, :, None, None]
        return self.model((pixels - mean) / std)


def native_loss(model, images, labels, epoch, stats):
    if stats is None:
        return training_loss(model, images, labels, epoch)
    mean = images.new_tensor(stats[0])[None, :, None, None]
    std = images.new_tensor(stats[1])[None, :, None, None]
    pixels = (images * std + mean).clamp(0, 1)
    return training_loss(NormalizedModel(model, stats), pixels, labels, epoch)
