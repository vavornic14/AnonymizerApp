from pydantic import BaseModel
from typing import List, Dict, Any
import torch
import torch.nn.functional as F


# --- Pydantic Models for API Requests/Responses ---

class AnonymizationRequest(BaseModel):
    """Request model for anonymizing text."""
    text: str


class AnonymizationResponse(BaseModel):
    """Response model for anonymized text and detected entities."""
    anonymized_text: str
    entities: List[Dict[str, Any]]


class DeanonymizationRequest(BaseModel):
    """Request model for deanonymizing text."""
    text: str


class DeanonymizationResponse(BaseModel):
    """Response model for deanonymized text."""
    original_text: str


# --- Anonymization Class ---

class Anonymizer:
    """
    A hybrid anonymization class that uses a fine-tuned NER model
    and a simple in-memory map for pseudonymization.
    """

    def __init__(self, tokenizer, model, id2label, label2id):
        self.tokenizer = tokenizer
        self.model = model
        self.id2label = id2label
        self.label2id = label2id
        self.anonymization_map = {}
        self.reverse_map = {}

    def _get_ner_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Performs NER inference on the text and returns a list of entities.
        NOTE: This is a simplified version for the demo.
        """
        if not self.model or not self.tokenizer:
            return []

        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = [self.id2label[p.item()] for p in predictions[0]]

        entities = []
        current_entity_label = None
        current_entity_text = ""
        current_entity_start = -1

        for i, (token, label) in enumerate(zip(tokens, labels)):
            if token.startswith("Ġ"):
                token = " " + token[1:]

            if label.startswith("B-"):
                if current_entity_text:
                    entities.append({
                        "text": current_entity_text.strip(),
                        "label": current_entity_label,
                        "start": current_entity_start,
                        "end": len(current_entity_text) + current_entity_start - 1,
                    })
                current_entity_label = label[2:]
                current_entity_text = token
                current_entity_start = i
            elif label.startswith("I-") and current_entity_label:
                current_entity_text += token
            else:
                if current_entity_text:
                    entities.append({
                        "text": current_entity_text.strip(),
                        "label": current_entity_label,
                        "start": current_entity_start,
                        "end": len(current_entity_text) + current_entity_start - 1,
                    })
                current_entity_label = None
                current_entity_text = ""
                current_entity_start = -1

        if current_entity_text:
            entities.append({
                "text": current_entity_text.strip(),
                "label": current_entity_label,
                "start": current_entity_start,
                "end": len(current_entity_text) + current_entity_start - 1,
            })

        return entities

    def anonymize_text(self, text: str) -> (str, List[Dict[str, Any]]):
        """
        Anonymizes text by replacing detected entities with placeholders.
        """
        entities = self._get_ner_entities(text)
        anonymized_text = text

        # Sort entities by start position to handle overlaps gracefully
        entities.sort(key=lambda x: x["start"], reverse=True)

        for entity in entities:
            original_text = entity["text"]
            replacement = f"[{entity['label']}]"
            anonymized_text = anonymized_text[:entity['start']] + replacement + anonymized_text[
                entity['start'] + len(original_text):]

            # Store the mapping for deanonymization
            if replacement not in self.anonymization_map:
                self.anonymization_map[replacement] = original_text
                self.reverse_map[original_text] = replacement

        return anonymized_text, entities

    def deanonymize_text(self, text: str) -> str:
        """
        Deanonymizes text by restoring original entities.
        """
        original_text = text
        for placeholder, original_value in self.anonymization_map.items():
            original_text = original_text.replace(placeholder, original_value)
        return original_text
