# References
# Web Framework - https://flask.palletsprojects.com/
# Cross-Origin Requests - https://flask-cors.readthedocs.io/
# Image processing - https://pillow.readthedocs.io/
# Model inference (PyTorch) - https://pytorch.org/docs/stable/index.html
# REST API - https://restfulapi.net/
import os
import io
import sys
import base64

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

from model import ViTClassifier
from transforms import val_transforms

app = Flask(__name__, static_folder="static")
CORS(app)

CLASS_NAMES = [
    "Proliferative",
    "Membranous Pattern",
    "Mesangial Hypercellularity",
    "Minimal Changes"
]

CLASS_DESCRIPTIONS = {
    "Proliferative":              "Focal or diffuse increase in glomerular cellularity, including crescentic forms.",
    "Membranous Pattern":         "Thickening of the glomerular basement membrane due to immune deposits.",
    "Mesangial Hypercellularity": "Expansion of mesangial cells and matrix within the glomerulus.",
    "Minimal Changes":            "Near-normal glomeruli on light microscopy; podocyte fusion on EM."
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

model = ViTClassifier(num_classes=len(CLASS_NAMES))
try:
    model.load_state_dict(torch.load("vit_kidney_classifier.pth", map_location=device))
    print("Weights loaded successfully.")
except FileNotFoundError:
    print("WARNING: vit_kidney_classifier.pth not found. Using random weights.")

model.to(device)
model.eval()


def predict_image(pil_image):
    img_rgb = pil_image.convert("RGB")
    x       = val_transforms(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(x)
        probs   = F.softmax(outputs, dim=1)[0]
    all_probs = {CLASS_NAMES[i]: round(probs[i].item() * 100, 2) for i in range(len(CLASS_NAMES))}
    pred_idx  = probs.argmax().item()
    return CLASS_NAMES[pred_idx], round(probs[pred_idx].item() * 100, 2), all_probs


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files   = request.files.getlist("files")
    results = []

    for f in files:
        if not f.filename:
            continue
        try:
            img = Image.open(f.stream)

            pred_class, confidence, all_probs = predict_image(img)

            # Thumbnail for display
            thumb = img.convert("RGB")
            thumb.thumbnail((400, 400))
            buf = io.BytesIO()
            thumb.save(buf, format="JPEG", quality=85)
            image_b64 = base64.b64encode(buf.getvalue()).decode()

            results.append({
                "filename":          f.filename,
                "predicted_class":   pred_class,
                "confidence":        confidence,
                "description":       CLASS_DESCRIPTIONS[pred_class],
                "all_probabilities": all_probs,
                "image_b64":         image_b64
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({"filename": f.filename, "error": str(e)})

    return jsonify(results)


if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True, port=5000)