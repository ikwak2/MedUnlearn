"""Official tiled UEraser objective; validation/test never call this function."""
import torch
import torch.nn.functional as F
from official_ueraser import adversarial_augmentation_loss, UEraser

AA_CONFIG = dict(name='UEraser', repeat=5, adversarial_epochs=50,
    later_repeat=1, jpeg=False, additional_pgd=False,
    implementation='official tiled per-image maximum-loss objective',
    source_commit='6076a275a521b5bcefd3ae4a698b7e6958e2ae7a')


def repeats(epoch):
    return 5 if epoch < 50 else 1


def training_loss(model, images, labels, epoch, augs=UEraser):
    count=repeats(epoch)
    captured=[]
    hook=model.register_forward_hook(lambda module,args,output:captured.append(output.detach()))
    try:
        loss=adversarial_augmentation_loss(model,images,labels,count,augs=augs)
    finally:
        hook.remove()
    assert len(captured)==1
    logits=captured[0]
    if count>1:
        losses=F.cross_entropy(logits,labels.tile(count),reduction='none').view(count,-1)
        chosen=losses.argmax(dim=0)
        logits=logits.view(count,len(labels),-1)[chosen,torch.arange(len(labels),device=labels.device)]
    return loss,logits
