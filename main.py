import io
import json
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from llm_service import generate_treatment_plan

app = FastAPI(title="Plant Disease Diagnostic API")

# Setup templates folder
templates = Jinja2Templates(directory="templates")

# --- 1. DEFINE CLASS LABELS (38 PlantVillage Classes) ---
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# --- 2. LOAD TRAINED MODEL ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def load_model():
    # EfficientNet-B0 backbone matching training setup
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(CLASS_NAMES))

    # Load model weights (fallbacks gracefully if weights file is not yet built)
    try:
        model.load_state_dict(
            torch.load("best_plantvillage_model.pth", map_location=device)
        )
        print("Successfully loaded 'best_plantvillage_model.pth'")
    except FileNotFoundError:
        print(
            "Warning: 'best_plantvillage_model.pth' not found. Running with uninitialized weights."
        )

    model.to(device)
    model.eval()
    return model


model = load_model()

# --- 3. IMAGE PREPROCESSING TRANSFORMS ---
img_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# --- 4. ENDPOINTS ---


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Renders the web UI frontend."""
    return templates.TemplateResponse(
        name="index.html", context={"request": request}, request=request
    )


@app.post("/api/predict-by-model")
async def predictByModel(file: UploadFile = File(...)):
    """Receives an image file, runs inference through PyTorch, and returns diagnosis."""
    # 1. Read uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # 2. Transform image
    input_tensor = img_transforms(image).unsqueeze(0).to(device)

    # 3. Model Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

        # Get top prediction
        top_prob, top_idx = torch.max(probabilities, dim=0)

        # Get top 3 predictions for breakdown
        top3_probs, top3_indices = torch.topk(probabilities, 3)

    top_class_raw = CLASS_NAMES[top_idx.item()]
    crop, disease = (
        top_class_raw.split("___")
        if "___" in top_class_raw
        else ("Plant", top_class_raw)
    )
    disease_formatted = disease.replace("_", " ")

    # 4. Prepare Response JSON
    response_data = {
        "crop": crop.replace("_", " "),
        "disease": disease_formatted,
        "is_healthy": "healthy" in disease.lower(),
        "confidence": round(top_prob.item() * 100, 2),
        "top_predictions": [
            {
                "label": CLASS_NAMES[idx.item()]
                .replace("___", " - ")
                .replace("_", " "),
                "confidence": round(prob.item() * 100, 2),
            }
            for prob, idx in zip(top3_probs, top3_indices)
        ],
    }

    return response_data


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Image preprocessing
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = img_transforms(image).unsqueeze(0).to(device)

    # 2. PyTorch Model Evaluation
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        top_prob, top_idx = torch.max(probabilities, dim=0)
        # Get top 3 predictions for breakdown
        top3_probs, top3_indices = torch.topk(probabilities, 3)

    top_class_raw = CLASS_NAMES[top_idx.item()]
    crop, disease = (
        top_class_raw.split("___")
        if "___" in top_class_raw
        else ("Plant", top_class_raw)
    )

    crop_clean = crop.replace("_", " ")
    disease_clean = disease.replace("_", " ")
    is_healthy = "healthy" in disease.lower()

    # 3. Dynamic LLM Treatment Generation
    treatment_plan = generate_treatment_plan(
        crop=crop_clean, disease=disease_clean, is_healthy=is_healthy
    )

    print(treatment_plan)

    return {
        "crop": crop_clean,
        "disease": disease_clean,
        "is_healthy": is_healthy,
        "confidence": round(top_prob.item() * 100, 2),
        "top_predictions": [
            {
                "label": CLASS_NAMES[idx.item()]
                .replace("___", " - ")
                .replace("_", " "),
                "confidence": round(prob.item() * 100, 2),
            }
            for prob, idx in zip(top3_probs, top3_indices)
        ],
        "treatment_plan": treatment_plan,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
