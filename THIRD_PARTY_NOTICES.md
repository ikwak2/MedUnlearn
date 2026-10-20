# Third-party sources

The repository's BSD3-Clause license applies to the authors' release code. Incorporated upstream sources retain their original copyright and license notices; it does not replace those notices.

| Component | Upstream and revision | Use |
|---|---|---|
| HYPO / Delusive Adversary | `https://github.com/TLMichael/Delusive-Adversary`, `8f033e000d85f93c34ed4a55f020952b240da235` | Linf steps, official CIFAR ResNet18, P1 VGG16 |
| AR | `https://github.com/psandovalsegura/autoregressive-poisoning`, `bf350b0cf2dc8ef4786a38950b994320dc450da6` | Official model and coefficient reference; the verified float32 recurrence is in `ar_generation.py` |
| T-AP | `https://github.com/lhfowl/adversarial_poisons`, `a1d58d8e31bcb01da494f39ad2b0a707c09acbf4` | Crafting/evaluation models and differentiable augmentation |
| GUE | `https://github.com/hong-xian/gue`, `d23ca8e15de2b1ff4e5301004d1b2522bfae2842` | BOME core, ResNet18 and UNet in `third_party/GUE` |
| Common EM/REM/SEM defenders | Authors' original Unlearn2 source snapshot | `third_party/Unlearn2`, original MIT notice retained |
| UEraser | `https://github.com/lafeat/ueraser`, `6076a275a521b5bcefd3ae4a698b7e6958e2ae7a` | Exact official per-image max-loss augmentation; fetched separately with `fetch-aa` |

Upstream license files are retained next to copied native sources where provided. Actual experiment source hashes and the explicitly limited portability edits are recorded in `configs/source_provenance.json`. Public data, model weights, and third-party datasets are not redistributed.
