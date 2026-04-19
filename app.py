from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

import torch
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)

# folder to store uploaded images
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ------------------ MODEL DEFINITIONS (required to load .pth) ------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.pool  = nn.MaxPool2d(2, 2)

        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.ups.append(DoubleConv(f * 2, f))

        self.final = nn.Conv2d(features[0], 1, 1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x    = self.ups[i](x)
            skip = skip_connections[i // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        return torch.sigmoid(self.final(x))


class StageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3),
        )
    def forward(self, mask_pred):
        return self.head(mask_pred)


class AMDStageModel(nn.Module):
    def __init__(self, unet, stage_head):
        super().__init__()
        self.unet       = unet
        self.stage_head = stage_head

    def forward(self, x):
        mask_pred   = self.unet(x)
        stage_logit = self.stage_head(mask_pred)
        return mask_pred, stage_logit


# ------------------ LOAD MODELS ------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STAGE_NAMES = {0: "Early AMD", 1: "Intermediate AMD", 2: "Advanced AMD"}

# Disease classifier (Keras) — untouched
classifier_model = load_model("models/amd_disease_model.keras")

# Stage model (PyTorch .pth)
_unet       = UNet(in_channels=1)
_stage_head = StageClassifier()
stage_model = AMDStageModel(_unet, _stage_head)
stage_model.load_state_dict(torch.load("models/amd_stage_best (1).pth", map_location=DEVICE))
stage_model.to(DEVICE)
stage_model.eval()

classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']


# ------------------ ML FUNCTIONS ------------------

def predict_disease(img_path):
    
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = img.reshape(1, 224, 224, 3)
    pred = classifier_model.predict(img)
    return classes[np.argmax(pred)]


def predict_stage_and_lesion(img_path, threshold=0.5):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")

    img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
    oct_tensor  = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    oct_tensor  = oct_tensor.to(DEVICE)

    with torch.no_grad():
        mask_pred, stage_logit = stage_model(oct_tensor)

    mask_np  = mask_pred[0, 0].cpu().numpy()
    bin_mask = (mask_np > threshold).astype(np.uint8)

  
    bright_pixels = np.sum(img_resized > 70)   # pixels that are actual tissue, not black background
    lesion_pixels = int(bin_mask.sum())

    if bright_pixels > 0:
        lesion_pct = round((lesion_pixels / bright_pixels) * 100.0, 2)
    else:
        lesion_pct = 0.0

   
    stage_probs = F.softmax(stage_logit, dim=1)[0].cpu().numpy()
    confidence  = float(stage_probs.max())
    stage_idx   = int(stage_logit.argmax(dim=1).item())

    if confidence <= 0.6:
        if lesion_pct < 5.0:    stage_idx = 0
        elif lesion_pct < 25.0: stage_idx = 1
        else:                   stage_idx = 2

   
    mask_vis     = (bin_mask * 255).astype(np.uint8)
    overlay_gray = cv2.resize(img, (256, 256))
    overlay_rgb  = np.stack([overlay_gray] * 3, axis=-1)
    overlay_rgb[bin_mask > 0] = [255, 100, 0]

    return STAGE_NAMES[stage_idx], lesion_pct, mask_vis, overlay_rgb



def calculate_visibility(disease, stage, lesion_pct): 
    visibility = round(100 * np.exp(-2.5 * (lesion_pct / 100)), 2)
    return visibility


def analyze_oct(img_path):
    disease = predict_disease(img_path)
    mask_vis = None
    overlay_rgb = None

    if disease == "NORMAL":
        severity    = "No AMD"
        lesion_pct  = 0.0

    elif disease == "CNV":
        severity, lesion_pct, mask_vis, overlay_rgb = predict_stage_and_lesion(img_path)
        if lesion_pct<10:
            severity="Intermediate AMD"
        else:
            severity = "Advanced AMD"

    elif disease == "DRUSEN":
        severity, lesion_pct, mask_vis, overlay_rgb = predict_stage_and_lesion(img_path)

    else:  # DME or other
        severity, lesion_pct, mask_vis, overlay_rgb = predict_stage_and_lesion(img_path)
        severity = "Non-AMD Disease"

    
    visibility=calculate_visibility(disease, severity, lesion_pct)
    return disease, severity, visibility, lesion_pct, mask_vis, overlay_rgb



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    if file:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        filepath=filepath.replace("\\","/")
        file.save(filepath)

        disease, stage, visibility, lesion_pct, mask_vis, overlay_rgb = analyze_oct(filepath)

        base = os.path.splitext(file.filename)[0]
        mask_path    = os.path.join(app.config["UPLOAD_FOLDER"], base + "_mask.png")
        overlay_path = os.path.join(app.config["UPLOAD_FOLDER"], base + "_overlay.png")

        if mask_vis is not None:
            cv2.imwrite(mask_path, mask_vis)
            cv2.imwrite(overlay_path, overlay_rgb)

        return render_template(
            "upload.html",
            disease=disease,
            stage=stage,
            visibility=str(visibility) + "%",
            image_path=filepath,
            lesion_percent=lesion_pct,
            mask_image_path=mask_path if mask_vis is not None else None,
            overlay_image_path=overlay_path if overlay_rgb is not None else None,
        )

    return redirect(url_for("upload"))


@app.route('/details')
def details():
    disease           = request.args.get("disease")
    stage             = request.args.get("stage")
    visibility        = request.args.get("visibility")
    image_path        = request.args.get("image_path")
    lesion_percent    = request.args.get("lesion_percent")
    mask_image_path   = request.args.get("mask_image_path")
    overlay_image_path = request.args.get("overlay_image_path")

    return render_template(
        "details.html",
        disease=disease,
        stage=stage,
        visibility=visibility,
        image_path=image_path,
        lesion_percent=lesion_percent,
        mask_image_path=mask_image_path,
        overlay_image_path=overlay_image_path,
    )

if __name__ == "__main__":
    app.run(debug=True)