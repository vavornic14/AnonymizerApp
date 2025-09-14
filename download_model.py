from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_NAME = "Finguys/acta-anonymizer-financial"

print(f"Downloading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Tokenizer downloaded successfully.")

print(f"Downloading model for {MODEL_NAME}...")
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
print("Model downloaded successfully.")

print("All files are now cached locally.")