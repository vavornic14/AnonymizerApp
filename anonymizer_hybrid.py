import re
from typing import List, Dict, Any
from pydantic import BaseModel


class Entity(BaseModel):
    start: int
    end: int
    text: str
    label: str
    replacement: str


class AnonymizationRequest(BaseModel):
    text: str


class AnonymizationResponse(BaseModel):
    anonymized_text: str
    entities: List[Entity]


class DeanonymizationRequest(BaseModel):
    text: str
    entities: List[Entity]


class DeanonymizationResponse(BaseModel):
    original_text: str


class Anonymizer:
    def __init__(self):
        # A simple, rule-based anonymization dictionary
        self.rules = {
            "NAMES": re.compile(r'\b(John Smith|Jane Doe|Michael Johnson)\b', re.IGNORECASE),
            "PHONE_NUMBER": re.compile(r'\b\d{10}\b'),  # Matches a 10-digit number
            "PERSONAL_CODE": re.compile(r'\b\d{13}\b'),  # Matches a 13-digit number
            "ID_CARD": re.compile(r'\b[A-Z]{2}\s\d{6,}\b'),  # Matches a 2-letter and 6-digit or more ID card
            "EMAIL": re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'),  # Matches email addresses
            "IBAN": re.compile(r'\b[A-Z]{2}\d{2}[A-Z\d]{1,30}\b'),  # Matches IBAN format
        }

    def anonymize_text(self, text: str) -> (str, List[Entity]):
        entities = []
        anonymized_text = text

        # Find all matches for all rules
        for label, pattern in self.rules.items():
            for match in pattern.finditer(text):
                entities.append({
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(0),
                    "label": label,
                    "replacement": f"[{label}]"
                })

        # Sort entities by start position in descending order
        sorted_entities = sorted(entities, key=lambda e: e["start"], reverse=True)

        # Perform replacements on the original text
        for entity in sorted_entities:
            start = entity["start"]
            end = entity["end"]
            replacement_text = entity["replacement"]
            anonymized_text = anonymized_text[:start] + replacement_text + anonymized_text[end:]

        # Correctly format entities to match the Pydantic model
        pydantic_entities = [
            Entity(
                start=e["start"],
                end=e["end"],
                text=e["text"],
                label=e["label"],
                replacement=e["replacement"]
            )
            for e in sorted_entities
        ]

        return anonymized_text, pydantic_entities

    def deanonymize_text(self, text: str, entities: List[Dict[str, Any]]) -> str:
        # Sort entities by start position in descending order to avoid messing up indices
        sorted_entities = sorted(entities, key=lambda e: e["start"], reverse=True)

        original_text = text
        for entity in sorted_entities:
            replacement_start = original_text.find(entity["replacement"])
            if replacement_start != -1:
                replacement_end = replacement_start + len(entity["replacement"])
                original_text = (
                        original_text[:replacement_start] +
                        entity["text"] +
                        original_text[replacement_end:]
                )

        return original_text
