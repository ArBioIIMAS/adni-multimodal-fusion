# Multimodal Neuroimaging Integration for Three-Class Alzheimer's Disease Staging

This repository contains the source code accompanying the manuscript *"Multimodal learning with MRI, PET, and DTI neuroimaging biomarkers for classifying subjects with Alzheimer's disease and mild cognitive impairment"*.

## Overview

We evaluate unimodal baselines alongside three multimodal data integration architectures—**early fusion**, **intermediate fusion**, and **late fusion**—for automated three-class staging across Cognitively Normal (**CN**), Mild Cognitive Impairment (**MCI**), and Alzheimer's Disease (**AD**) subjects.

### Key Pipeline Components
1. **Biomarker Representation (228 features across 6 subcortical regions of interest):**
   * **sMRI:** 3D morphological shape descriptors extracted following FreeSurfer v6.0 segmentation.
   * **FDG-PET:** Intensity distribution metrics normalized to MNI152 space (SPM12).
   * **DTI:** Microstructural diffusion distribution metrics.
2. **Leak-Free Validation Protocol:**
   * Strict subject-level partitioning evaluated over **5 independent outer repetitions**.
   * Stratified **6-fold cross-validation** on the training partition for hyperparameter optimization of latent representation models (intermediate fusion).
   * Stratified **6-fold cross-validation** on the training partition for downstream classifier tuning (`ElasticNet`, `KNN`, `SVM`, `Random Forest`, `CatBoost`, and `MLP`).
3. **Model Interpretability:**
   * Feature importance and consensus ranking analysis across unimodal and early fusion schemes.

---

## Repository Structure

```text
├── Unimodals_and_late_fusion/    # Unimodal baselines, late fusion meta-models, and weight optimization
├── Early_fusion/                 # Feature-level concatenation and feature importance evaluation
├── Intermediate_fusion/          # Latent representation learning (SPCA, DL, DLO, SC, AE, DDL)
└── README.md
```

