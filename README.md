# Pediatric caries robustness with YOLOv8

This repository provides analysis scripts and summary results for the study:

**Robustness and clinical triage reliability of smartphone-based deep learning for pediatric dental caries screening under real-world image degradation**

## Overview

This study evaluated the robustness of a lightweight YOLOv8n detector for pediatric dental caries screening from smartphone-based oral images. The original ICDAS-based labels were reorganized into four clinically relevant categories:

* no caries
* mild caries
* moderate caries
* severe caries

The model was evaluated on an independent clean test set and under simulated image degradation conditions, including low light, overexposure, Gaussian blur, motion blur, JPEG compression, contrast shift, color shift, and occlusion. Image-level clinical triage performance was also evaluated for any caries, moderate-or-severe caries, and severe caries.

## Repository contents

```text
scripts/
  convert_7cls_to_4cls.py
  augment_train_4cls_to_6000.py
  make_degraded_test_4cls.py
  run_degraded_val_4cls.py
  run_degraded_val_4cls_perclass.py
  run_image_level_triage_4cls.py
  plot_robustness_4cls.py
  make_plos_figures.py

results/
  robustness_summary_4cls.csv
  perclass_robustness_4cls.csv
  severe_caries_robustness.csv
  image_level_triage_all_conditions.csv
  image_level_triage_main_conditions.csv

figures/
  Fig1_overall_robustness.png
  Fig2_severe_caries_robustness.png
```

## Data availability

The original pediatric oral image dataset is not publicly released because it contains clinical image data from pediatric participants and is subject to institutional ethics and privacy restrictions. The repository provides de-identified aggregate evaluation results, robustness summaries, figure-generation outputs, and analysis scripts.

Access to the original image dataset may be requested from the corresponding author and requires approval from the relevant ethics committee and affiliated institution.

## Main results

The clean test set contained 187 images. On the independent clean test set, the YOLOv8n-based four-class model achieved:

* Precision: 0.570
* Recall: 0.660
* mAP50: 0.618
* mAP50-95: 0.382

For severe caries detection, the model achieved:

* Precision: 0.695
* Recall: 0.718
* mAP50: 0.756
* mAP50-95: 0.464

Image-level severe caries triage on the clean test set achieved:

* Sensitivity: 0.757
* Specificity: 0.953
* Precision: 0.800
* NPV: 0.941
* F1 score: 0.778
* Accuracy: 0.914

## Software environment

The experiments were conducted using:

* Python 3.8
* PyTorch 2.1.0
* Ultralytics YOLOv8.2.18
* OpenCV
* NumPy
* Pandas
* Matplotlib

## Notes

The scripts contain example paths used in the original local environment. Users should update the dataset paths before running the scripts.

The repository does not include the original pediatric oral images, raw labels, trained weights, or qualitative examples containing clinical images.
