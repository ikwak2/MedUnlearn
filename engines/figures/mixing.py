"""Fixed stratified nested masks; only training image selection changes."""
import numpy as np
import torch


def make_masks(labels,ratios=(20,40,60,80),seed=42):
    labels=np.asarray(labels);classes=np.unique(labels)
    rng=np.random.default_rng(seed)
    order={c:rng.permutation(np.flatnonzero(labels==c)) for c in classes}
    counts=np.array([len(order[c]) for c in classes],dtype=np.int64)
    masks={};previous=np.zeros(len(labels),dtype=bool);details={}
    for ratio in ratios:
        numerator=counts*ratio
        quotas=numerator//100
        total=(len(labels)*ratio+50)//100
        remaining=total-int(quotas.sum())
        # Largest remainder; class-index order resolves equal remainders.
        ranking=np.argsort(-(numerator%100),kind='stable')
        quotas[ranking[:remaining]]+=1
        mask=np.zeros(len(labels),dtype=bool)
        for c,n in zip(classes,quotas):mask[order[c][:n]]=True
        assert int(mask.sum())==total
        assert not np.any(previous & ~mask),'Non-nested allocation; inspect class supports'
        masks[ratio]=mask;previous=mask
        details[ratio]=dict(target_percent=ratio,actual_percent=100*total/len(labels),
            total_n=len(labels),poison_n=total,
            classes=[dict(class_index=int(c),train_n=int(n),poison_n=int(q)) for c,n,q in zip(classes,counts,quotas)])
    return masks,details


def mixed_class(original,mask):
    class MixedDataset(original):
        def __init__(self,dataset,split,poison=None):
            super().__init__(dataset,split,poison)
            self.mixing_mask=mask if split=='train' and poison is not None else None
            if self.mixing_mask is not None:assert len(mask)==len(self.rows)

        def __getitem__(self,index):
            if self.mixing_mask is None or self.mixing_mask[index]:
                return super().__getitem__(index)
            # Match original CachedDataset's uint8 conversion exactly.
            value=torch.from_numpy(np.array(self.x[index],copy=True)).float()
            if self.x.dtype==np.uint8:value=value/255
            return value,int(self.rows[index]['class_index']),index
    return MixedDataset
