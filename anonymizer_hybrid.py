from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import regex as re
import random


class AnonymizationRequest(BaseModel):
    text: str


class Entity(BaseModel):
    start: int
    end: int
    text: str
    label: str
    replacement: str


class AnonymizationResponse(BaseModel):
    anonymized_text: str
    entities: List[Entity]


class DeanonymizationRequest(BaseModel):
    text: str


class DeanonymizationResponse(BaseModel):
    original_text: str


class Anonymizer:
    def __init__(self, tokenizer: Optional[AutoTokenizer] = None,
                 model: Optional[AutoModelForTokenClassification] = None, id2label: Optional[Dict[int, str]] = None,
                 label2id: Optional[Dict[str, int]] = None):
        if tokenizer and model:
            self.nlp_pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple"
            )
        else:
            self.nlp_pipeline = None

        self.regex_rules = {
            "PHONE": re.compile(r'\b(07\d{8}|06\d{7}|02\d{8}|03\d{8})\b'),
            "CNP": re.compile(r'\b(1|2)\d{12}\b'),
            "ID_CARD": re.compile(r'[A-Z]{2}\s?\d{7}'),
            "IBAN": re.compile(r'\bRO\d{2}[A-Z]{4}\d{16}\b'),
            "EMAIL": re.compile(r'[\w\.-]+@[\w\.-]+\.\w{2,}'),
            "ADDRESS": re.compile(r'strada\s[\w\s]+\d+,\s?[\w\s]+')
        }

    def _generate_label_replacement(self, label: str) -> str:
        return f"[{label}]"

    def _filter_overlapping_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Sort by length (descending) to prioritize longer entities
        entities.sort(key=lambda x: x['end'] - x['start'], reverse=True)

        filtered_entities = []

        # Keep track of the parts of the text that have been covered by an entity
        covered_indices = set()

        for entity in entities:
            entity_range = range(entity['start'], entity['end'])
            is_overlapping = False

            # Check if this entity's range overlaps with any previously covered range
            for i in entity_range:
                if i in covered_indices:
                    is_overlapping = True
                    break

            if not is_overlapping:
                filtered_entities.append(entity)
                # Add all indices of the current entity to the covered set
                for i in entity_range:
                    covered_indices.add(i)

        return filtered_entities

    def anonymize_text(self, text: str):
        entities_to_anonymize = []

        # 1. Collect entities from the NER model
        if self.nlp_pipeline:
            ner_results = self.nlp_pipeline(text)
            for entity in ner_results:
                label = entity['entity_group']
                if label not in ["O"]:
                    entities_to_anonymize.append({
                        'start': entity['start'],
                        'end': entity['end'],
                        'original_text': entity['word'],
                        'label': label,
                        'replacement': self._generate_label_replacement(label)
                    })

        # 2. Collect entities from regex rules
        for label, pattern in self.regex_rules.items():
            for match in pattern.finditer(text):
                entities_to_anonymize.append({
                    'start': match.span()[0],
                    'end': match.span()[1],
                    'original_text': match.group(0),
                    'label': label,
                    'replacement': self._generate_label_replacement(label)
                })

        # 3. Filter overlapping entities
        non_overlapping_entities = self._filter_overlapping_entities(entities_to_anonymize)

        # 4. Sort entities by end position in descending order for right-to-left replacement
        non_overlapping_entities.sort(key=lambda x: x['end'], reverse=True)

        # 5. Perform replacements from right to left
        anonymized_text = text
        final_entities = []
        for entity in non_overlapping_entities:
            start = entity['start']
            end = entity['end']
            replacement = entity['replacement']

            anonymized_text = anonymized_text[:start] + replacement + anonymized_text[end:]

            final_entities.append(Entity(
                start=start,
                end=start + len(replacement),
                text=entity['original_text'],
                label=entity['label'],
                replacement=replacement
            ))

        return anonymized_text, sorted(final_entities, key=lambda e: e.start)

    def deanonymize_text(self, anonymized_text: str):
        # Deanonymization is not fully implemented in this demo as it's typically irreversible.
        # This function serves as a placeholder for a complete system.
        return anonymized_text
