# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union
import torch
import numpy as np
from .model_loader import load_model_and_tokenizer, get_device # Relative import

# Initialize FastAPI app
app = FastAPI(title="Greek Legal RoBERTa Embedder")

# --- Pydantic Models for Request and Response ---
class TextItem(BaseModel):
    text: str

class EmbeddingRequest(BaseModel):
    instances: List[TextItem] # Vertex AI expects an 'instances' key
    # You can add 'parameters' if needed, e.g., for choosing pooling strategy
    # parameters: Optional[dict] = None

class EmbeddingPrediction(BaseModel):
    embedding: List[float]

class EmbeddingResponse(BaseModel):
    predictions: List[EmbeddingPrediction] # Vertex AI expects a 'predictions' key

# --- Load Model and Tokenizer on Startup ---
# This will be executed once when Uvicorn starts the app
tokenizer, model = None, None

@app.on_event("startup")
async def startup_event():
    global tokenizer, model
    print("Application startup: Loading model...")
    tokenizer, model = load_model_and_tokenizer()
    print(f"Model ready on device: {get_device()}")

# --- Health Check Endpoint (Good practice for Vertex AI) ---
@app.get("/health", status_code=200)
async def health_check():
    # Basic check, can be expanded (e.g., check model readiness)
    return {"status": "healthy"}

# --- Prediction Endpoint (Vertex AI expects /predict) ---
@app.post("/predict", response_model=EmbeddingResponse)
async def predict_embeddings(request: EmbeddingRequest):
    global tokenizer, model
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please try again shortly.")

    texts_to_embed = [item.text for item in request.instances]
    if not texts_to_embed:
        raise HTTPException(status_code=400, detail="No text instances provided.")

    device = get_device()
    try:
        inputs = tokenizer(
            texts_to_embed,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Using mean pooling of last hidden states
            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()

        predictions_list = []
        for emb_array in batch_embeddings:
            predictions_list.append(EmbeddingPrediction(embedding=emb_array.tolist()))

        return EmbeddingResponse(predictions=predictions_list)

    except Exception as e:
        print(f"Error during embedding generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # For local testing with Uvicorn (not directly used by Docker image build)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
