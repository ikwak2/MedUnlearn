# Reported results snapshot

Updated: 2026-10-22T08:36:44+09:00. Final Table 3: **48/48 UE conditions**, each with its matched CLEAN.

- `comparison.csv`, `classwise.csv`, and `aggregates.json`: prescribed main selections and anonymous confusion matrices.
- `selection_sensitivity.csv`: selection-labelled paper-aligned results, including P5 CIFAR validation-best versus final.
- `figures56/`: Figure 5 **144/192** fresh ratio conditions plus 96 existing endpoints; Figure 6 **54/72** training conditions including shared common CLEAN+AA and separate native controls. Incomplete conditions remain gaps. CSV rows with `primary=False` are sensitivity results.
- `auxiliary/`: **32/32** common-method SSIM conditions and **20/21** independent common generation costs. Paper-aligned P1/P5/AR/T-AP auxiliary measurements are not included; these partial tables are not the complete final Tables 4/5.
- `analysis/`: class recall decreases, confusion matrices, and No Finding/disease decomposition regenerated from the main snapshot.

Native CIFAR configurations differ from the common protocol. Raw scores do not isolate algorithm effects. MRI patient independence remains unverified. No patient images, identifiers, individual predictions or checkpoints are distributed.
