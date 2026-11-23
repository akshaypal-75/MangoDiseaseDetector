# Mango Disease Detector 🥭

A patch-based CNN application for mango leaf disease classification, based on MobileNetV2 transfer learning, patch-wise inference, and Grad-CAM explainability.

## Attribution
This repository contains code from the original project by **Tarunj Gupta**, reused with permission. The original project is available at https://github.com/tarunjgupta/MangoDiseaseDetector. This repository is maintained by **Akshay Pal**.

## Features
- MobileNetV2 transfer learning
- Patch-based predictions
- Grad-CAM explainability
- Streamlit web interface
- Command-line prediction

## Project structure
```
app.py
convert_model.py
dataset_loader.py
gradcam.py
model.py
patch_utils.py
predict.py
train.py
requirements.txt
saved_model/class_names.txt
```

## Setup
Python 3.10 is recommended for the pinned TensorFlow version.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place training images in `dataset/` with one subfolder per class: `Anthracnose`, `Bacterial_Canker`, `Scab`, and `Healthy`.

Train the model:
```bash
python train.py --dataset ./dataset --epochs 10
```

Run the web app:
```bash
streamlit run app.py
```

The trained model file is not included in this repository; train locally to generate `saved_model/mango_model.h5`.

## Classes
Anthracnose · Bacterial Canker · Scab · Healthy

## Notes
Predictions are experimental and should not replace professional agricultural diagnosis.
