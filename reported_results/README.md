# Reported results snapshot

Updated: 2026-10-23T18:56:38+09:00. Final Table 3: **48/48 UE conditions**, each with its matched CLEAN.

- `comparison.csv`, `classwise.csv`, and `aggregates.json`: prescribed main selections and anonymous confusion matrices.
- `selection_sensitivity.csv`: selection-labelled paper-aligned results, including P5 CIFAR validation-best versus final.
- `figures56/`: Figure 5 **192/192** fresh ratio conditions plus 96 existing endpoints; Figure 6 **72/72** training conditions including shared common CLEAN+AA and separate native controls. All planned conditions are complete. CSV rows with `primary=False` are supplementary checkpoint results.
- `auxiliary/`: complete final-main SSIM **48/48** and independent generation costs **33/33**. The same paper-aligned/common method routing as Table 3 is used. LSP variants share one generator.
- `analysis/`: class recall decreases, confusion matrices, and No Finding/disease decomposition regenerated from the main snapshot.

Native CIFAR configurations differ from the common protocol. Raw scores do not isolate algorithm effects. MRI patient independence remains unverified. No patient images, identifiers, individual predictions or checkpoints are distributed.
