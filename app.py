from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image
import io
import os
import numpy as np

# Model classes
from transformers import SwinForImageClassification, ViTForImageClassification
from torchvision.models import resnet50

# Using our exact 49 classes
CLASS_NAMES = [
    'WNL', 'AH', 'AION', 'ARMD', 'BRVO', 'CB', 'CF', 'CL', 'ME', 'NV', 'CRAO', 'CRS',
    'CRVO', 'CSC', 'CWS', 'DN', 'DR', 'EX', 'ERM', 'GRT', 'HPED', 'HTR', 'HR', 'IIH',
    'LS', 'MCA', 'MH', 'MHL', 'MS', 'MYA', 'ODC', 'ODE', 'ODP', 'ON', 'ODPM', 'PRH',
    'RD', 'RHL', 'RTR', 'RP', 'RPEC', 'RS', 'RT', 'SOFE', 'ST', 'TD', 'TSLN', 'TV', 'VS'
]

app = FastAPI(title="IRIS - Medical Inference Ensemble")

# Enable CORS for local testing if frontend is detached
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static directory for the single-page frontend
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL LOADING LOGIC ---
# We keep these globally to load them only once into VRAM at startup.
models = {
    "swin": {"path": "swin_model_multilabel.pth", "instance": None},
    "vit": {"path": "vit_model_multilabel.pth", "instance": None},
    "resnet50": {"path": "cnn_resnet50_multilabel.pth", "instance": None}
}

def load_swin():
    model = SwinForImageClassification.from_pretrained(
        "microsoft/swin-tiny-patch4-window7-224", ignore_mismatched_sizes=True, num_labels=49)
    if os.path.exists(models["swin"]["path"]):
        model.load_state_dict(torch.load(models["swin"]["path"], map_location=device))
        model.to(device).eval()
        return model
    return None

def load_vit():
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224", ignore_mismatched_sizes=True, num_labels=49)
    if os.path.exists(models["vit"]["path"]):
        model.load_state_dict(torch.load(models["vit"]["path"], map_location=device))
        model.to(device).eval()
        return model
    return None

def load_resnet50():
    model = resnet50()
    in_features = model.fc.in_features
    model.fc = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(in_features, 49))
    if os.path.exists(models["resnet50"]["path"]):
        model.load_state_dict(torch.load(models["resnet50"]["path"], map_location=device))
        model.to(device).eval()
        return model
    return None

@app.on_event("startup")
async def startup_event():
    print(f"Server starting. Allocating models to {device}...")
    models["swin"]["instance"] = load_swin()
    models["vit"]["instance"] = load_vit()
    models["resnet50"]["instance"] = load_resnet50()
    
    loaded = sum(1 for m in models.values() if m["instance"] is not None)
    print(f"Loaded {loaded}/3 models successfully.")

# --- INFERENCE TRANSFORM ---
# Exactly matching the validation/test transforms required by the architectures
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """ Serves the main UI file. """
    with open(os.path.join("static", "index.html"), "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """ Main inference endpoint handling uploaded images """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

    input_tensor = inference_transform(image).unsqueeze(0).to(device)

    results = {}
    ensemble_probs = np.zeros(49)
    active_models = 0

    with torch.no_grad():
        for model_name, config in models.items():
            model = config["instance"]
            if model is None:
                results[model_name] = {"status": "Not Trained / Weights Missing", "top_predictions": []}
                continue
            
            # Predict
            if model_name in ["swin", "vit"]:
                logits = model(input_tensor).logits
            else:
                logits = model(input_tensor)
                
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            # Record for ensemble average
            ensemble_probs += probs
            active_models += 1
            
            # Extract top 5 for the individual model
            top5_idx = np.argsort(probs)[-5:][::-1]
            top_predictions = [{"class": CLASS_NAMES[i], "probability": float(probs[i])} for i in top5_idx]
            
            results[model_name] = {
                "status": "Success",
                "top_predictions": top_predictions
            }

    # Calculate ensemble average
    if active_models > 0:
        ensemble_probs /= active_models
        top5_ens_idx = np.argsort(ensemble_probs)[-5:][::-1]
        results["ensemble"] = {
            "status": f"Averaged across {active_models} models",
            "top_predictions": [{"class": CLASS_NAMES[i], "probability": float(ensemble_probs[i])} for i in top5_ens_idx]
        }
    else:
        results["ensemble"] = {
            "status": "No models trained yet to form an ensemble. Please train models first.",
            "top_predictions": []
        }

    return JSONResponse(content=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
