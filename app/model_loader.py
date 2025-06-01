# app/model_loader.py
from transformers import AutoTokenizer, AutoModel
import torch
import os

#MODEL_NAME = "AI-team-UoA/GreekLegalRoBERTa_v2"
MODEL_NAME = "AI-team-UoA/GreekLegalRoBERTa_v4"
# Determine device (CPU or GPU if available and configured in Docker)
# For Vertex AI custom containers, you'll configure GPU usage in the deployment.
# Here, we'll default to CPU for simplicity and broader Docker image compatibility.
# If your Docker base image has CUDA and you provision GPU on Vertex AI, this will pick it up.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = None
model = None

def load_model_and_tokenizer():
    global tokenizer, model
    if tokenizer is None or model is None:
        print(f"Loading tokenizer for {MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print(f"Loading model {MODEL_NAME} to {DEVICE}...")
        model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
        model.eval() # Set to evaluation mode
        print("Model and tokenizer loaded successfully.")
    return tokenizer, model

def get_device():
    return DEVICE

if __name__ == "__main__":
    # For testing the loader directly
    load_model_and_tokenizer()
    print(f"Running on device: {get_device()}")
