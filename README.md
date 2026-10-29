# FER2013 Facial Expression Classification

Facial expression classification experiments by **Bilge Evleksiz**, using PyTorch, transfer learning and the FER2013 dataset.

This repository presents my existing notebook experiments with **ResNet34**, **ResNet50** and a **Vision Transformer (ViT)**. It preserves the original training code and saved training/evaluation outputs. The project is shared as an experimental study, with its evaluation limitations documented below.

## Task and data

The task is to classify cropped face images into seven expression categories:

`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.

The original notebook reads a local folder-format copy of FER2013 through Google Colab/Google Drive. Saved logs show 28,709 training images and initially 7,178 evaluation images. Later ViT reports contain 7,177 images; another cell records removal of one unreadable image. The exact original download provider/version and execution order are not recorded.

FER2013 consists of 48×48 grayscale face images. Reference: [original FER challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data). A commonly used folder-format distribution is available [on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013); this link does not establish which copy was used in these experiments. Obtain data separately and review the chosen provider's terms.

The notebook uses `ImageFolder` alphabetical class IDs. These differ from the numeric code order in the original FER CSV format; raw CSV labels should not be interpreted using the folder order without an explicit mapping.

## Methods

### ResNet experiments

- ImageNet-initialized ResNet34 (`IMAGENET1K_V1`) and ResNet50 (`IMAGENET1K_V2`).
- Replace the final fully connected layer with a seven-output layer.
- Convert grayscale images to three channels and resize to 224×224.
- Training augmentation: horizontal flipping, rotation and random resized cropping.
- Fine tune all model parameters using Adam, learning rate `1e-4`, and cross entropy loss.
- Reduce learning rate on a validation-loss plateau; save the minimum-loss checkpoint and use early stopping.

The original ResNet transforms use `ToTensor` without ImageNet mean/std normalization. This documentation-only release preserves that code and does not claim that it implements a corrected preprocessing pipeline.

### ViT experiments

- `timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=7)`.
- Two augmentation configurations, 224×224 inputs and ImageNet mean/std normalization.
- AdamW, cosine learning-rate scheduling, mixed precision and loss-based early stopping.
- Saved classification reports with per-class precision, recall, F1 and support.

The exact `timm` version and pretrained-weight revision were not retained. The setup cells also reference a third-party [Emotion-recognition repository](https://github.com/omar178/Emotion-recognition); cloning or listing its models is not evidence that those models produced the ResNet/ViT results below.

## Saved experimental results

These values were read from the notebook's **existing saved text outputs**. They were not generated or independently reproduced during documentation preparation.

| Experiment | Original notebook cell (1-based) | Saved result |
| --- | --- | --- |
| ResNet34 | 20 | Minimum-loss checkpoint at epoch 7: loss **0.8767**, accuracy **69.02%** |
| ResNet50 | 17 | Minimum-loss checkpoint at epoch 6: loss **0.9022**, accuracy **68.10%** |
| ViT, first augmentation configuration | 23 | Printed classification report: accuracy **0.67**, macro F1 **0.66**, support **7,177** |
| ViT, second augmentation configuration | 24 | Printed classification report: accuracy **0.69**, macro F1 **0.66**, support **7,177** |

**Evaluation limitation:** the directory named `test` was used during training for checkpoint selection and early stopping. These are therefore model-selection/validation results, **not independent held-out test performance**. Comparing these scores does not establish a general ranking of the architectures. The ViT reports are rounded to two decimal places and must not be attributed to ResNet.

The supplied notebook does not contain trained checkpoint files, per-image prediction records or a confusion matrix. A historical confusion matrix cannot be reconstructed from the aggregate reports alone.

## Repository contents

```text
README.md
fer2013_experiments.ipynb
requirements.txt
.gitignore
```

The notebook is the original experimental record. Its training code and training/evaluation outputs are preserved. Only two directory-listing outputs were cleared because they exposed unrelated personal filenames; notebook metadata was reduced to standard kernel information. No training algorithm, data split, preprocessing, checkpoint logic or metric calculation was changed. No new experiment was run.

The repository contains no dataset, trained weights, third-party repository copies or added training pipeline. It does not provide a ready-to-use pretrained inference model.

## Inspecting or reproducing the notebook

The notebook can be viewed directly on GitHub; no execution is needed to inspect its code and saved output.

For a future reproduction attempt:

1. Open `fer2013_experiments.ipynb` in Google Colab. The original saved logs report CUDA/Tesla T4; a compatible GPU environment is recommended for training.
2. Install the core experiment dependencies from `requirements.txt`. This file lists needed libraries, not an exact reconstruction of the original environment.
3. Provide your own FER2013 copy with `train/<class>` and `test/<class>` directories.
4. Update the notebook's `/content` and Google Drive paths for your environment. The notebook contains alternative experiments and setup/debug cells; it is not a verified run-all pipeline.
5. Review the known issues below before executing. Reproducing the original notebook does not correct its evaluation design.

The initial setup cells also install TensorFlow, Keras, OpenCV and Optuna. They are not required by the later PyTorch ResNet/ViT training code and are not included in the core dependency file. Accessing Google Drive uses Colab's own integration.

## Known limitations and future work

- Introduce a separate validation split for model selection and reserve an appropriate test set for final reporting. Historical test reuse must still be disclosed.
- Replace the original corrupt-image fallback: it reads the next valid image, which can duplicate a sample during evaluation. A later cleanup cell permanently deletes unreadable files; review it before execution.
- Verify the actual dataset, class distribution, duplicate/identity overlap and source terms. This was not possible from saved logs alone.
- Use consistent, recorded preprocessing and model/weight versions. Changing preprocessing requires a separately documented experiment, not attaching the old score to new code.
- The final ViT cell depends on `classification_report_eval` defined in the previous ViT cell. Some setup/debug cells assume files or directories already exist.
- Class imbalance is visible in the saved evaluation reports (for example, disgust has 111 examples and happy has 1,773). Training class counts and weighting/sampling experiments were not verified here.
- No independent test score, runtime benchmark, calibrated confidence, real-time webcam system, face detector or music-recommendation integration is demonstrated by this notebook.

## Data and third-party materials

Data and pretrained assets must be obtained from their respective providers under applicable terms. The original dataset provenance and all asset redistribution permissions are not established by this notebook. No project-wide open-source license is assigned in this documentation release. Third-party model implementations and assets retain their own terms; no third-party source or weights are bundled.
