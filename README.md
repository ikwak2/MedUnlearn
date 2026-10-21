# MedUnlearn

Reproduction code for the revised medical-imaging unlearnable-example benchmark. The final **Table 3 policy** is fixed in [`configs/table3.json`](configs/table3.json):

| Setting | Methods | Checkpoint reported |
|---|---|---|
| Paper-aligned | HYPO P1, HYPO P5 | Prespecified validation-best |
| Paper-aligned | AR, T-AP | Fixed final epoch |
| Common benchmark | EM, REM, SEM, GUE, LSPNormal, LSPAUG, UT-AP, HYPO P3 | Maximum clean-validation balanced accuracy; earliest tie |

The datasets are **CIFAR-10, X-ray(fixed), MRI4, and MRI15**. X-ray(fixed) is the patient-disjoint, single-annotation, 12-class subset with ordinary cross entropy. The historical first-label 15-class X-ray benchmark and weighted-CE ablations are excluded.

Each method is compared with its **own matching CLEAN control**. LSPAUG uses CLEAN_AUG. Paper-aligned training differs from the common benchmark in optimizer, augmentation, epochs, seed, and the CIFAR training/holdout protocol. Report paired CLEAN/UE metrics and recall decreases rather than treating the raw scores as an algorithm-only ranking. P5 CIFAR validation-best and final-epoch scores differ substantially; both selection-labelled results are retained.

## Inspect the protocol and regenerate reported analyses

These commands do not require a GPU or access to images:

```bash
python -m medunlearn plan
python -m medunlearn plan --dataset MRI4 --method HYPO_P1
python -m unittest discover -s tests -v
python -m pip install -r requirements/analysis.txt
python -m medunlearn analyze --reported --output outputs/reported_analysis
```

Run commands from this repository root, or install with `python -m pip install .` to use the `medunlearn` command elsewhere. The numerical environments use Python3.10; Python3.12+ is excluded because the native PyTorch2.0.1 environment is required. [`docs/VALIDATION.md`](docs/VALIDATION.md) records the checks actually performed.

[`reported_results/`](reported_results/README.md) contains the **48/48 completed UE conditions**, their matched CLEAN metrics, and anonymous class aggregates. X-ray(fixed) GUE is now included: **45.60% accuracy, 10.63% balanced accuracy**, selected at epoch25. The accompanying Figure5/6 snapshot contains completed conditions only; gaps remain for running experiments. This repository does not distribute images, patient/sample identifiers, individual predictions, trained checkpoints, server queues, or incident logs.

[`reported_results/selection_sensitivity.csv`](reported_results/selection_sensitivity.csv) retains both selection-labelled native results. The [class analyses](reported_results/analysis/) and [Figure5/6 previews](reported_results/figures56/) were regenerated from the anonymous snapshot. [`reported_results/auxiliary/`](reported_results/auxiliary/) contains completed common-method SSIM and generation-cost measurements; paper-aligned P1/P5/AR/T-AP measurements are outside that auxiliary snapshot, so these are partial final-protocol tables.

## Prepare the data

Obtain the datasets under their original access terms and follow [`docs/DATA.md`](docs/DATA.md). Preserve the original dataset versions and train/test directory membership used by the paper.

```bash
# Use the common environment for split construction and image caches.
python -m venv .venv-common
.venv-common/bin/pip install -r requirements/common.txt
.venv-common/bin/python -m medunlearn prepare \
  --data-root /absolute/path/to/datasets --workspace /absolute/path/to/workspace
```

An already authorized local final-manifest directory can instead be imported with `import-manifests`. Manifests and generated images remain in the workspace, outside this repository. A different source cohort can produce different counts; the X-ray preparation command checks the published 8,843/3,758/8,417 split sizes. Exact cohort reconstruction requires the original source memberships, which are not distributed here.

## Run Table 3

The numerical environments used by the two protocol families are kept separate:

```bash
python -m venv .venv-reference
.venv-reference/bin/pip install -r requirements/reference.txt

# Paper-aligned P1/P5/AR/T-AP: PyTorch 2.0.1 / torchvision 0.15.2.
CUDA_VISIBLE_DEVICES=0 .venv-reference/bin/python -m medunlearn run \
  --dataset MRI4 --method HYPO_P1 --workspace /absolute/path/to/workspace

# Common methods: PyTorch 2.5.1+cu121 / torchvision 0.20.1+cu121.
CUDA_VISIBLE_DEVICES=0 .venv-common/bin/python -m medunlearn run \
  --dataset MRI4 --method EM --workspace /absolute/path/to/workspace

# GUE uses the official 2.0.1 generator and the common downstream trainer.
CUDA_VISIBLE_DEVICES=0 .venv-common/bin/python -m medunlearn run \
  --dataset MRI4 --method GUE --workspace /absolute/path/to/workspace \
  --generation-python /absolute/path/to/.venv-reference/bin/python

.venv-common/bin/python -m medunlearn summarize --workspace /absolute/path/to/workspace
.venv-common/bin/python -m medunlearn analyze \
  --results /absolute/path/to/workspace/results --output outputs/class_analysis
```

`run` trains or reuses the matching CLEAN, creates the selected method's poison, trains the downstream model, and evaluates the prescribed checkpoint. Test predictions are produced after training. A completed condition is reused; changed code/configuration requires a new workspace. Use `run ... --dry-run` to inspect routing without loading data or CUDA.

## Tables 4–5 and Figure4

The same method-routing policy applies to perturbation measurements. SSIM uses the effective training representation, including the native P1/P5 uint8 quantization and training-index subset. Generation profiling excludes existing CLEAN/crafting pretraining and downstream training. Use the reference environment for native methods/GUE generation and the common environment for the other generators. Native CPU SSIM additionally uses `requirements/reference-analysis.txt`.

```bash
.venv-common/bin/python -m medunlearn ssim --dataset MRI4 --method EM \
  --workspace /absolute/path/to/workspace --output outputs/ssim_MRI4_EM.json \
  --examples outputs/perturbation_MRI4_EM
CUDA_VISIBLE_DEVICES=0 .venv-reference/bin/python -m medunlearn profile --dataset MRI4 --method HYPO_P1 \
  --workspace /absolute/path/to/workspace --output outputs/cost_MRI4_P1.json
CUDA_VISIBLE_DEVICES=0 .venv-common/bin/python -m medunlearn tsne --dataset MRI4 \
  --workspace /absolute/path/to/workspace --output outputs/tsne_MRI4
```

The t-SNE command uses the new CLEAN/REM checkpoints on the same complete clean test set, saves the512-dimensional features privately, and reproduces PCA50→t-SNE with seeds42/0/1. Its panels are qualitative and use separate coordinate fits. Profile outputs record GPU/software and PyTorch memory metrics; measurements on different hardware do not form one controlled cost table.
The optional SSIM examples produce clean/poisoned/magnified-difference panels for Figures2–3 using the training example nearest the median SSIM loss; the displayed difference scale and clipping are explicit. Keep image outputs outside the public source repository.

## Figures 5 and 6

Finish the corresponding Table 3 condition first. Use the same environment as that method's downstream trainer.

```bash
python -m medunlearn fetch-aa  # exact pinned official UEraser source

# Figure 5: fixed generator; nested class-stratified 20/40/60/80% training subsets.
CUDA_VISIBLE_DEVICES=0 .venv-reference/bin/python -m medunlearn figure --mode ratio --ratio 20 \
  --dataset MRI4 --method HYPO_P1 --workspace /absolute/path/to/workspace

# Figure 6: fresh matching CLEAN+AA followed by UE+AA.
CUDA_VISIBLE_DEVICES=0 .venv-reference/bin/python -m medunlearn figure --mode aa \
  --dataset MRI4 --method HYPO_P1 --workspace /absolute/path/to/workspace

.venv-common/bin/python -m medunlearn summarize-figures \
  --workspace /absolute/path/to/workspace --output outputs/figures56
```

The figure collector preserves all selection-labelled CSV results and renders ACC/BA/macro-F1 curves/bars with each method's prescribed primary selection. Unfinished conditions remain gaps or missing bars.

Figure 5 reuses Table 3 CLEAN/UE as its 0/100% endpoints. Figure 6 applies official per-image maximum-loss UEraser: K=5 for the first 50 epochs, then K=1, without JPEG or extra PGD. Native base augmentation and input normalization are preserved. See [`docs/PROTOCOLS.md`](docs/PROTOCOLS.md) for the native P1/P5 CIFAR poisoned-holdout detail, paired controls, model-selection rules, perturbation budgets, and limitations.

## Source provenance and validation

`engines/reference/` preserves the actual frozen sources used for the four paper-aligned methods. Portability edits change the CIFAR root and host metadata and normalize trailing whitespace; numerical training logic is retained. `engines/common/` contains the final common trainer and selected generators; unused CLEAN-search candidates and the superseded GUE generator were removed. `configs/source_provenance.json` records original/release source hashes. Third-party notices are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

The published runs use one seed per protocol. MRI patient-level independence could not be verified. Low classification accuracy is not a formal privacy guarantee, and poor clean disease recall limits claims about additional protection. GPU/software changes can prevent bitwise identical training.
