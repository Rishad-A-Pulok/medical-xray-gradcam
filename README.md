# Explainable Pneumonia Classification from Chest X-rays using Grad-CAM

A reproducible educational project comparing a small CNN with ImageNet-pretrained ResNet18 for binary pediatric chest X-ray classification. The ResNet18 model also supports Grad-CAM visualization.

> **Research-only.** This is not a diagnostic system. Grad-CAM highlights model sensitivity, not verified disease regions.

## Dataset

Download [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), based on Kermany et al. (2018), after checking its license. Extract into `data/chest_xray/` such that `data/chest_xray/train/NORMAL/`, `train/PNEUMONIA/`, `val/`, and `test/` exist. Do not commit image data to GitHub.

The original validation split is tiny (often 16 images). Our script combines the original train and validation pools and creates a reproducible 85/15 stratified development split. It keeps the original test set held out, checks exact duplicate hashes across the test boundary, and removes exact duplicates within the two pools. This is **not** a patient-level split: available filenames may not reliably identify patients, and near-duplicates are not detected.

## Run (Google Colab or local Python)

```bash
pip install -r requirements.txt
python -m src.pipeline prepare --data-root data/chest_xray
python -m src.pipeline train --model cnn --epochs 8
python -m src.pipeline train --model resnet18 --epochs 8
python -m src.pipeline evaluate --model cnn
python -m src.pipeline evaluate --model resnet18
python -m src.pipeline gradcam --model resnet18 --image data/chest_xray/test/PNEUMONIA/EXAMPLE.jpeg
```

Replace `EXAMPLE.jpeg` with a real image. Colab users should upload/clone this repository and mount Drive or download the Kaggle data into the expected folder. Enable GPU under Runtime settings if available.

## Evaluation

Report accuracy, per-class precision/recall/F1, macro F1, confusion matrix, ROC-AUC and average precision. Evaluate both models on the same untouched test manifest; select training settings using validation only. Files are written to `results/` after actual runs. No experimental results are claimed in this starter repository.

## Grad-CAM

The script backpropagates the chosen class logit through the final ResNet18 residual block and overlays a normalized heatmap on the X-ray. Inspect true positives, true negatives, false positives, and false negatives, including possible reliance on text or borders. Do not interpret heatmaps as segmentation masks.

## Limitations

Single pediatric source; class imbalance; no verified patient-independent splitting; potential label noise, shortcut learning, and domain shift. Further research should include patient-level metadata, external evaluation, confidence intervals, calibration, and clinical expert review.

## References

- Kermany et al., *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*, Cell (2018).
- Selvaraju et al., *Grad-CAM: Visual Explanations From Deep Networks via Gradient-Based Localization*, ICCV (2017).
- [Torchvision ResNet18 documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)
