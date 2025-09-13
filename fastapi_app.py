from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from transformers import AutoTokenizer, AutoModelForTokenClassification
from anonymizer_hybrid import Anonymizer, AnonymizationRequest, AnonymizationResponse, DeanonymizationRequest, \
    DeanonymizationResponse
import os
import torch

app = FastAPI(title="FinTech PrivacyGuard Demo")

# Load pre-trained NER model and tokenizer
# This model has been updated to the one you asked about.
MODEL_NAME = "EvanD/xlm-roberta-base-romanian-ner-ronec"
tokenizer = None
model = None
id2label = None
label2id = None

try:
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    id2label = model.config.id2label
    label2id = {v: k for k, v in id2label.items()}
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model from Hugging Face: {e}")
    print("The app will run in a limited mode without model-based anonymization.")
    tokenizer = None
    model = None

# Initialize the anonymizer. It will use a dummy model if loading failed.
anonymizer = Anonymizer(tokenizer=tokenizer, model=model, id2label=id2label, label2id=label2id)


# --- FastAPI Endpoints ---

@app.get("/")
async def serve_index():
    """Serves the main HTML page for the demo."""
    file_path = "static/index.html"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404,
                            detail="Frontend file not found. Please ensure 'static/index.html' exists.")
    return FileResponse(file_path)


@app.post("/anonymize", response_model=AnonymizationResponse)
async def anonymize_text_endpoint(request: AnonymizationRequest):
    """Anonymizes the input text and returns entities and the anonymized version."""
    try:
        if not tokenizer or not model:
            anonymized_text = request.text
            entities = []
            return AnonymizationResponse(anonymized_text=anonymized_text, entities=entities)

        anonymized_text, entities = anonymizer.anonymize_text(request.text)
        return AnonymizationResponse(anonymized_text=anonymized_text, entities=entities)
    except Exception as e:
        print(f"Anonymization Error: {e}")
        raise HTTPException(status_code=500, detail=f"Anonymization failed: {str(e)}")


@app.post("/deanonymize", response_model=DeanonymizationResponse)
async def deanonymize_text_endpoint(request: DeanonymizationRequest):
    """Deanonymizes the input text."""
    try:
        original_text = anonymizer.deanonymize_text(request.text)
        return DeanonymizationResponse(original_text=original_text)
    except Exception as e:
        print(f"Deanonymization Error: {e}")
        raise HTTPException(status_code=500, detail=f"Deanonymization failed: {str(e)}")


# Mount static files to serve the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn

    if not os.path.exists("static"):
        print("Warning: 'static' directory not found. The app will fail to serve the frontend.")
        print("Please create a 'static' folder and place 'index.html' inside it.")
    uvicorn.run(app, host="127.0.0.1", port=8000)
