# Data layout and cohort reconstruction

The data root must contain the original paper source layout:

```text
datasets/
  mri/train/<class>/*                 # MRI4
  mri/test/<class>/*
  mri_15/train/<class>/*              # MRI15
  mri_15/test/<class>/*
  xray/train/<image files>
  xray/test/<image files>
  xray/train.csv                     # original source membership: Image,label
  xray/test.csv                      # original source membership: Image,label
  xray/Data_Entry_2017.csv            # original NIH metadata
  cifar-10-batches-py/{batches.meta,data_batch_1,...,data_batch_5,test_batch}
```

The MRI folders/classes and NIH source membership must be the same versions used in the paper. The repository provides the full split/filter algorithm, not private membership lists. Changing source train/test membership, image versions, order, or labels defines a different benchmark. Published class supports are in `reported_results/class_composition_*.csv`.

Preparation removes generated `_noisy` demos, excludes training images that duplicate test content or share known test patients, groups identical training content and known patients, and chooses the closest class-balanced 70/30 train/validation allocation among 256 GroupShuffleSplit candidates with seed42. Original test membership is retained. The MRI labels `_NORMAL` and `NORMAL` are canonicalized as in the actual runs. MRI grouping handles exact duplicates; it cannot establish patient independence without patient identifiers.

For NIH X-ray, the original grouping is constructed first. All three splits are then filtered using **original Finding Labels with exactly one annotation**. Multi-label records are not reduced to their first label in the final dataset. Classes with fewer than50 retained **training** images are excluded without reshuffling patients: Edema, Hernia, and Pneumonia. The remaining12 classes are Atelectasis, Cardiomegaly, Consolidation, Effusion, Emphysema, Fibrosis, Infiltration, Mass, No Finding, Nodule, Pleural_Thickening, and Pneumothorax. Single annotation does not establish absence of other clinical conditions.

Medical images are converted to RGB and bilinearly resized to256×256; CIFAR remains32×32. Cache arrays are uint8 NCHW and cache metadata is tied to the split manifest. All generation and normalization use the declared training inputs. Native HYPO's CIFAR recipe intentionally includes its separate reference holdout in generation, matching that protocol; this is not the medical validation procedure.

To use an authorized local final-manifest set, organize its `MRI4`, `MRI15`, `CIFAR-10`, and `X-ray` directories with `train.csv`, `validation.csv`, `test.csv`, and `classes.json`, plus `xray_fixed_audit.json`, then run:

```bash
python -m medunlearn import-manifests --manifests /absolute/path/to/final_manifests \
  --data-root /absolute/path/to/datasets --workspace /absolute/path/to/workspace
```

Manifest `path` fields must point to the locally authorized raw files; importing preserves their contents and hashes. No data are downloaded automatically.
