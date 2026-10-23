# Validation performed for this release

- The final policy resolves48UE conditions:16paper-aligned and32common. The historical15-class X-ray label is rejected by the entry point. The snapshot now contains all48completedUE conditions, including X-ray(fixed) GUE.
- Six policy/metric tests passed, including imbalanced-class BA, exact CLEAN/UE sample pairing, and aggregate consistency.
- The common MRI4 cache loaded with3,817/1,632/1,311samples. A real ResNet18 CPU forward/backward passed using the staged public common engine.
- In the actual PyTorch2.0.1 reference image, all four public native model paths passed CPU forward/backward. Their AA source derivations loaded, and the pixel-space normalization plus official AA objective produced finite gradients.
- The same frozen numerical native engines passed actual RTX4090 medical batch32 CUDA checks in the running revision experiments: plain peak allocated3,519,539,200bytes; K5AA8,095,549,440bytes. The public entry points additionally remove the machine-specific GPU-name restriction and parameterize filesystem/host metadata.
- An analysis environment installed from`requirements/analysis.txt` regenerated the reported48-condition heatmaps, confusion matrices, and No Finding analysis. The SSIM entry point returned approximatelyzero loss for identical effective clean/poison arrays.
- The result exporter checked all48 confusion matrices against the main ACC/BA/macro-F1, matched CLEAN metrics, class support, and selected epoch. Figure5/6 previews were regenerated from the mixed native/common CSV snapshot; non-primary selection rows and AA rows are excluded from ratio curves.
- The public X-ray(fixed) ordinary-CE benchmark source is byte-identical to the production benchmark: SHA256`6f2de6ed1bf107706bc74834fa0865576d9d7887349d00c97836448802bf6124`.
- Wheel build and installed-package protocol lookup are checked separately from running in the source checkout. Code compilation and Git whitespace checks passed.

This packaging validation does not repeat all48long GPU experiments. Original experiments and additional Figure5/6 runs are preserved in their private runtimes; the repository contains their selected code and anonymous aggregates. GPU/platform changes, original native cuDNN behavior, and incomplete source cohort membership can affect exact numerical reproduction. Refer to`docs/DATA.md` before claiming the published cohort has been reconstructed.

## Completed snapshot and minimal-source refresh (2026-10-23)

- Verified all 48 final main conditions, 192 intermediate ratio trainings, 72 AA/control trainings, 48 SSIM conditions and 33 independent cost measurements. Earlier common P1/P5/AR/T-AP rows and the legacy 15-class X-ray data are excluded from the public snapshot.
- Six tests pass, including exact completed ratio coverage, final auxiliary method routing, native CIFAR SSIM training-only counts/quantization, and all release source hashes.
- After removing unused common generator branches, LSPNormal, LSPAUG, P3 and UT-AP produced bit-identical arrays to their previous public sources on deterministic synthetic CPU fixtures. No medical images or new long GPU runs were used. The used GUE core and seed function ASTs are unchanged.
- Rebuilt the wheel after pruning the unused GUE architectures, CLI demos and duplicate data utility. Full long-training numerical equivalence across platforms is not claimed by these packaging checks.
