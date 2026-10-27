# Blood Cell Classification using ViT and Score-CAM

A PyTorch implementation for classifying cropped blood-cell images into **RBC**, **WBC**, and **Platelets** using a Vision Transformer (ViT) trained **from scratch**, with Score-CAM visual explanations.

> **Research note:** The 99.18% test accuracy below is a previously reported result from the project, not a result reproduced by these newly added scripts. Re-run evaluation using your exact split and checkpoint before treating it as reproduced.

## Reported experiment

- Architecture: ViT-Base / Patch16 / 224 × 224
- Pretrained weights: none (`pretrained=False`)
- Training: 50 epochs, augmentation and noise
- Reported test accuracy: 99.18%
- Reported cropped patches: 4,153 RBC, 361 Platelets, 372 WBC
- Intended split: 70% train / 15% validation / 15% test

### Important evaluation caution
Blood-cell crops from the same original smear image can be highly correlated. If patches from one source image occur in both train and test sets, evaluation may be optimistic. For a stronger estimate, split by original source image before cropping, or group all crops from each source image into a single split.

## Repository layout

- `train.py` — train ViT from scratch on folders of images
- `evaluate.py` — evaluate a saved checkpoint and write a classification report and confusion matrix
- `predict.py` — predict one image
- `scorecam.py` — Score-CAM heatmap generation for a trained ViT
- `requirements.txt` — Python dependencies

## Dataset setup

Arrange the images in ImageFolder format (folder names become class names):

```text
data/
  train/
    Platelets/
    RBC/
    WBC/
  val/
    Platelets/
    RBC/
    WBC/
  test/
    Platelets/
    RBC/
    WBC/
```

Use your own BCCD-derived crops and respect the original dataset's terms and attribution requirements. Do not commit the full dataset unless its license permits redistribution.

## Install

Python 3.10 or 3.11 is recommended. Install a PyTorch build appropriate for your machine from the official PyTorch selector, then:

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --data-dir data --output-dir outputs --epochs 50 --batch-size 16
```

This expects `data/train` and `data/val`. The best checkpoint is saved to `outputs/best_model.pth`; class names are saved alongside it. Adjust batch size to fit your GPU memory.

## Evaluate

```bash
python evaluate.py --data-dir data/test --checkpoint outputs/best_model.pth --class-names outputs/classes.json --output-dir outputs/evaluation
```

The script writes `classification_report.txt`, `confusion_matrix.png`, and `predictions.csv`.

## Predict one image

```bash
python predict.py --image path/to/cell.png --checkpoint outputs/best_model.pth --class-names outputs/classes.json
```

## Score-CAM

```bash
python scorecam.py --image path/to/cell.png --checkpoint outputs/best_model.pth --class-names outputs/classes.json --output outputs/scorecam.png
```

Score-CAM is a visual explanation, not proof that the model uses biologically meaningful features. Inspect examples across all classes, including errors.

## Reproducibility and limitations

The scripts provide a reproducible baseline pipeline, but the original training notebook, exact preprocessing, split manifest, trained checkpoint, and original Score-CAM figures were not available during this repository update. Results can differ until those exact artifacts and settings are restored. Never publish a metric as reproduced unless you rerun it on the documented test split.

## Citation / attribution

Add the exact source URL and license for the BCCD dataset version used in your project before public release. Cite the Vision Transformer and Score-CAM papers in any academic report or presentation.
