# app/model_loader.py
from transformers import AutoTokenizer, AutoModel
import torch
import os
 
# Get model name from environment variable, with a default
DEFAULT_MODEL_NAME = "AI-team-UoA/GreekLegalRoBERTa_v3" # Default if not set in env
MODEL_NAME = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
 
# Determine device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
_tokenizer = None
_model = None
 
def load_model_and_tokenizer():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        print(f"Loading tokenizer for {MODEL_NAME}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print(f"Loading model {MODEL_NAME} to {DEVICE}...")
        _model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
        _model.eval() # Set to evaluation mode
        print("Model and tokenizer loaded successfully.")
    return _tokenizer, _model
 
def get_tokenizer_and_model():
    # Ensures model is loaded if not already
    if _tokenizer is None or _model is None:
        load_model_and_tokenizer()
    return _tokenizer, _model
 
def get_device():
    return DEVICE
 
def get_model_name():
    return MODEL_NAME

if __name__ == "__main__":
    # For testing the loader directly
    load_model_and_tokenizer()
    print(f"Running on device: {get_device()}")
    print(f"Using model: {get_model_name()}")
