# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union, Optional
import torch
import numpy as np
from .model_loader import get_tokenizer_and_model, get_device, get_model_name # Updated import
import traceback # For detailed error logging

# Initialize FastAPI app
app = FastAPI(title="Greek Legal RoBERTa Embedder")

# --- Pydantic Models for OpenAI-compatible Request and Response ---
class OpenAIEmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str # Client specifies model, we'll use it for logging/validation if needed
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None # OpenAI supports this, we can just accept it

class OpenAIEmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class OpenAIUsageData(BaseModel):
    prompt_tokens: int
    total_tokens: int # For embeddings, usually same as prompt_tokens

class OpenAIEmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[OpenAIEmbeddingData]
    model: str # The actual model name used by this server
    usage: OpenAIUsageData


@app.on_event("startup")
async def startup_event():
    print("Application startup: Pre-loading model...")
    get_tokenizer_and_model() # This will call load_model_and_tokenizer internally
    print(f"Model ready on device: {get_device()} using model: {get_model_name()}")

# --- Health Check Endpoint (Good practice for Vertex AI) ---
@app.get("/health", status_code=200)
async def health_check():
    tokenizer, model = get_tokenizer_and_model()
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "healthy"}

# --- OpenAI-Compatible Embedding Endpoint ---
@app.post("/v1/embeddings", response_model=OpenAIEmbeddingResponse)
async def create_embeddings_openai(request: OpenAIEmbeddingRequest):
    tokenizer, model = get_tokenizer_and_model()
    device = get_device()
    actual_model_name = get_model_name()

    effective_max_length = model.config.max_position_embeddings

    if request.encoding_format and request.encoding_format != "float":
        raise HTTPException(status_code=400, detail="Unsupported encoding_format. Only 'float' is supported.")

    model_output_dim = model.config.hidden_size
    if request.dimensions is not None:
        if request.dimensions != model_output_dim:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid 'dimensions' parameter. This model produces embeddings of dimension {model_output_dim}, but {request.dimensions} was requested."
            )
        # If dimensions match, it's fine. If the model supported variable dimensions, logic would go here.

    if isinstance(request.input, str):
        texts_to_embed = [request.input]
    elif isinstance(request.input, list):
        texts_to_embed = request.input
    else:
        raise HTTPException(status_code=400, detail="'input' must be a string or a list of strings.")

    if not texts_to_embed or not all(isinstance(t, str) for t in texts_to_embed):
        raise HTTPException(status_code=400, detail="Input must be a non-empty string or a list of non-empty strings.")
    if any(not t.strip() for t in texts_to_embed): # Check for empty or whitespace-only strings
        raise HTTPException(status_code=400, detail="Input strings cannot be empty or whitespace only.")

    # Calculate prompt_tokens
    total_prompt_tokens = 0
    try:
        for text_item in texts_to_embed:
            item_token_ids = tokenizer.encode(
                text_item,
                add_special_tokens=True,
                truncation=True,
                max_length=effective_max_length
            )
            total_prompt_tokens += len(item_token_ids)
    except Exception as e:
        print(f"Error during token counting: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during token counting: {str(e)}")

    try:
        # Batch tokenization for model inference
        inputs = tokenizer(
            texts_to_embed,
            padding=True,
            truncation=True,
            max_length=effective_max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Using mean pooling of last hidden states
            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            batch_embeddings_np = (sum_embeddings / sum_mask).cpu().numpy()

        openai_data_list = []
        for i, emb_array in enumerate(batch_embeddings_np):
            openai_data_list.append(OpenAIEmbeddingData(embedding=emb_array.tolist(), index=i))

        return OpenAIEmbeddingResponse(
            data=openai_data_list,
            model=actual_model_name, # Report the model actually used
            usage=OpenAIUsageData(prompt_tokens=total_prompt_tokens, total_tokens=total_prompt_tokens)
        )

    except Exception as e:
        print(f"Error during OpenAI-compatible embedding generation: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    # For local testing with Uvicorn (not directly used by Docker image build)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
