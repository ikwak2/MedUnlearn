# Final Tables 4 and 5

All measurements use the final Table 3 routing: paper-aligned P1/P5/AR/T-AP, common settings for the remaining methods. Display name X-ray denotes the final 12-class subset; its internal key is `X-ray(fixed)`.

## Table 4: Mean SSIM loss

| Method | CIFAR-10 | X-ray | MRI4 | MRI15 |
|---|---:|---:|---:|---:|
| EM | 0.04692 | 0.24921 | 0.25700 | 0.26126 |
| REM | 0.03311 | 0.06840 | 0.15532 | 0.14583 |
| SEM | 0.04104 | 0.08152 | 0.16628 | 0.17947 |
| HYPO P1 | 0.05709 | 0.09295 | 0.22025 | 0.25690 |
| HYPO P3 | 0.01106 | 0.00031 | 0.00283 | 0.00110 |
| HYPO P5 | 0.03044 | 0.16585 | 0.14421 | 0.16216 |
| AR | 0.02080 | 0.00123 | 0.00598 | 0.00510 |
| T-AP | 0.05543 | 0.15283 | 0.23600 | 0.24510 |
| UT-AP | 0.06569 | 0.23049 | 0.26768 | 0.30896 |
| LSP | 0.02300 | 0.10696 | 0.14539 | 0.15205 |
| GUE | 0.02223 | 0.03405 | 0.14568 | 0.15614 |

48 method/dataset conditions, with the LSP variants sharing perturbations. The actual training subsets are used; native P1/P5 include the loader's float32-to-uint8 truncation. CIFAR native P1/P5 use 49,000 training indices and exclude the poisoned holdout. Gaussian window 11, sigma 1.5, data range 1, population covariance, float64.

## Table 5: Generation cost

Each cell is **minutes / PyTorch peak allocated MiB**.

| Method | X-ray | MRI4 | MRI15 |
|---|---:|---:|---:|
| EM | 10.9 / 1,179.7 | 10.7 / 1,179.6 | 10.5 / 1,179.7 |
| REM | 541.8 / 1,371.8 | 502.4 / 1,371.7 | 498.8 / 1,371.8 |
| SEM | 64.9 / 1,335.8 | 63.4 / 1,335.7 | 62.2 / 1,335.8 |
| HYPO P1 | 33.7 / 14,003.4 | 14.7 / 14,003.3 | 8.4 / 14,003.4 |
| HYPO P3 | 1.8 / 1,018.5 | 0.7 / 1,018.4 | 1.9 / 1,018.5 |
| HYPO P5 | 0.4 / 27.0 | 0.2 / 9.0 | 0.1 / 36.0 |
| AR | 0.6 / 0.0 | 0.3 / 0.0 | 0.2 / 0.0 |
| T-AP | 148.7 / 3,417.0 | 65.9 / 3,416.9 | 36.7 / 3,417.0 |
| UT-AP | 96.4 / 1,028.0 | 41.7 / 1,027.9 | 24.0 / 1,028.0 |
| LSP | 0.5 / 0.0 | 0.3 / 0.0 | 0.2 / 0.0 |
| GUE | 2038.8 / 14,824.5 | 1055.7 / 7,662.4 | 518.8 / 14,823.7 |

33 independent full-generation measurements on RTX 4090 at 450 W, one run per condition. Existing CLEAN/crafting pretraining and downstream training are excluded. Native generators and GUE use PyTorch 2.0.1; other common generators use 2.5.1. CPU-only generation can have zero PyTorch allocated GPU memory; this is not total process memory. LSP variants share one cost measurement. These profiling runs do not replace the poison arrays used for performance or SSIM.

Portable commands: `medunlearn ssim` and `medunlearn profile`; see the root README. Machine assignments, queue settings and runtime logs are excluded from this release.
