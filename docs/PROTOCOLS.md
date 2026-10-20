# Final experimental protocols

| Method family | Native CIFAR | Medical extension | Primary selection |
|---|---|---|---|
| P1/P5 | 49,000 train / 1,000 holdout,150epochs,SGD .1,momentum .9,wd .0005,batch128,milestones100/125,crop+flip,no normalization | Existing corrected splits,256px,batch32,150epochs,SGD and crop+flip | Validation-best; medical clean-val BA, CIFAR augmented poisoned holdout for UE |
| AR | 50,000 train,no holdout,100epochs,SGD .1,batch128,StepLR50,crop+flip,official normalization | Existing corrected splits,256px,batch32,100epochs,train-only normalization | Fixed-final100 |
| T-AP | Separate40epoch crafting model; downstream200epochs,SGD .1,cosine,crop+flip | Existing corrected splits,256px,batch32; same40/200epoch structure | Fixed-final200 |
| Common medical | — | Adam .001,batch32,100epochs,no augmentation,ordinary CE | Highest clean validation BA; earliest tie |
| Common CIFAR | 35,000 train / 15,000 validation,SGD .1,momentum .9,Nesterov,wd .0005,batch128,200epochs,warmup5+cosine,train-only normalization,no augmentation | — | Highest clean validation BA; earliest tie |
| Common LSPAUG | Adam .001,100epochs,crop+flip | Adam .001,100epochs,crop+flip | Matched CLEAN_AUG, validation BA |

Native configurations retain seed0 and their original cuDNN settings; common downstream training uses seed42. Native P1 uses a separate VGG16 crafting model, with adaptive final pooling for medical256px input. P5 is class-specific Uniform[-8/255,8/255] noise. They are Linf methods in the adopted main protocol, unlike the old L2 variants that are excluded here.

GUE uses the official BOME core at commit `d23ca8e15de2b1ff4e5301004d1b2522bfae2842`,50generation epochs,seed3,Linf8/255,lr_cls .01,lr_atk .1,lr_tmp .0001,10inner iterations,eta1.5. Final generator selection has no validation/test access. Generation batch sizes are256/CIFAR,4/MRI4,8/MRI15,8/X-ray(fixed); downstream training uses the common protocol. Historical GUE implementations are excluded.

The common EM/REM/SEM use5000surrogate steps and Linf8/255, the existing REM/SEM defender implementation, and EOT5 for REM/SEM. Common UT-AP uses the validation-selected common CLEAN crafting model and250steps×8restarts. For CIFAR common P3/UT-AP, the Adam CLEAN crafting model is distinct from the SGD downstream CLEAN; the runner preserves this distinction. LSPNormal/LSPAUG share their generated perturbations; P3 keeps the declared common L2=.5 recipe. Parameter/norm choices are method-specific and must be reported as such.

The recorded common CIFAR REM run resumed a preserved generator checkpoint with a fresh deterministic data order after a machine interruption. MRI4 GUE also migrated hardware while retaining its generator state. The release supports those checkpoint formats, but a fresh run follows the declared protocol rather than recreating the exact interruption history. The native protocols also retain their original cuDNN behavior. Do not claim bitwise identity from a seed alone.

## Figure5

The generator and full poison array stay fixed. Training seed is the method's original seed; mask seed is42. Class-stratified largest-remainder allocation gives exact global fractions and nested20/40/60/80% masks. Only selected training images are replaced; retained clean images keep the same labels. Original CLEAN/UE results supply0/100% endpoints.

For native P1/P5 CIFAR, the49,000training images determine the ratio. At nonzero ratios the original fully poisoned, augmented1,000image holdout remains fixed, preserving the native checkpoint rule. The0% CLEAN control has its original clean holdout. This validation difference must be stated; fixed-final results are also retained as sensitivity analyses. Medical validation/test images remain clean.

## Figure6

Train a fresh protocol-matched CLEAN+AA before UE+AA. Preserve optimizer,base augmentation,epochs,model,normalization,seed,and checkpoint rule. Official UEraser uses per-image maximum CE over K=5augmentations for epochs1–50 and K=1thereafter; no JPEG or extraPGD. AA operates in pixel space, followed by the native input normalization. Validation/test evaluation does not invoke AA. Compare against the new CLEAN+AA and report recovery relative to the original UE.

## Reporting

ACC is the proportion of correctly classified test images. Class recall is TP/support. BA is the unweighted mean recall over all declared classes; no zero-recall class is dropped. Macro-F1 averages class F1. Recall decrease is matched CLEAN minus UE in percentage points. No Finding/11disease analyses retain12-way exact-label correctness; they are not separate binary training tasks. Confusion matrices are row-normalized, and prediction histograms show true support alongside prediction share.

Primary rules were fixed before the relevant test evaluations. Do not choose whichever validation-best/final row has the lower test score. In particular, native P5 CIFAR validation-best10.43% versus final83.71% must be disclosed with both selection-labelled outputs.
