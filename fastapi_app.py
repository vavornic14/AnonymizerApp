import os
import time
import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any
import requests
import cohere
import re

# Initialize the FastAPI app
app = FastAPI()

# Mount the static directory to serve static files (like index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")


# Anonymization/Deanonymization logic
class Anonymizer:
    def anonymize_text(self, text: str):
        entities = []
        # Example regex for different types of financial/personal data
        patterns = {
            "PHONE_NUMBER": r"(\b\d{10,12}\b)",
            "IBAN": r"\b[A-Z]{2}[0-9A-Z]{2}[0-9A-Z]{12,30}\b",
            "EMAIL_ADDRESS": r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
            "ID_CARD": r"\b[A-Z]{2}\s?\d{6,7}\b",
            "PERSONAL_CODE": r"\b\d{13}\b",
            "NAME": r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b"
        }

        # Collect all matches and sort them in reverse order to avoid index issues
        matches = []
        for entity_type, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                matches.append({
                    "entity_type": entity_type,
                    "original_text": match.group(0),
                    "start": match.start(),
                    "end": match.end()
                })

        matches.sort(key=lambda x: x['start'], reverse=True)

        anonymized_text = text
        for i, match in enumerate(matches):
            unique_id = f"[{match['entity_type']}_{i}]"
            anonymized_text = anonymized_text[:match['start']] + unique_id + anonymized_text[match['end']:]

            # Store entities in the correct dictionary format for the response
            entities.append({
                "unique_id": unique_id,
                "original_text": match['original_text'],
                "entity_type": match['entity_type']
            })

        return anonymized_text, entities

    def deanonymize_text(self, anonymized_text: str, entities: List[Dict[str, str]]):
        deanonymized_text = anonymized_text
        for entity in entities:
            unique_id = entity["unique_id"]
            original_text = entity["original_text"]
            deanonymized_text = deanonymized_text.replace(unique_id, original_text)
        return deanonymized_text


anonymizer = Anonymizer()


# Cohere API logic with exponential backoff
async def get_cohere_analysis(text: str):
    api_key = "XwLPqRhafzxz1bOdpLOlaCRvRqzVbKlpknGBrRxV"
    if not api_key:
        raise HTTPException(status_code=500, detail="COHERE_API_KEY environment variable not set.")

    co = cohere.AsyncClient(api_key)
    retries = 5
    delay = 1

    for i in range(retries):
        try:
            response = await co.chat(
                model='command-r',
                message=f"Analyze the following financial text and provide a concise summary, highlighting any potential risks or key details related to financial transactions. The text has been anonymized, do not mention the original details. Focus on a high-level overview. Text: {text}",
                temperature=0.7
            )
            return response.text
        except cohere.core.api_error.ApiError as e:
            if e.status_code == 429 and i < retries - 1:
                print(f"Attempt {i + 1} failed with 429 Too Many Requests. Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                delay *= 2
            else:
                raise HTTPException(status_code=e.status_code, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Cohere API call failed: {str(e)}")


# Pydantic models for request body
class AnonymizeRequest(BaseModel):
    text: str


class AnonymizeResponse(BaseModel):
    anonymized_text: str
    entities: List[Dict[str, str]]


class DeanonymizeRequest(BaseModel):
    text: str
    entities: List[Dict[str, str]]


class DeanonymizeResponse(BaseModel):
    original_text: str


class CohereAnalyzeResponse(BaseModel):
    analysis: str


# Endpoints
@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def redirect_to_docs():
    return "/static/index.html"


@app.post("/api/anonymize", response_model=AnonymizeResponse)
async def anonymize_text_endpoint(request: AnonymizeRequest):
    if not isinstance(request.text, str):
        raise HTTPException(status_code=400, detail="Input 'text' must be a string.")

    anonymized_text, entities_list = anonymizer.anonymize_text(request.text)

    # Check if anonymizer_text and entities_map are the correct types
    if not isinstance(anonymized_text, str) or not isinstance(entities_list, list):
        raise HTTPException(status_code=500, detail="Anonymizer function returned incorrect data types.")

    return {"anonymized_text": anonymized_text, "entities": entities_list}


@app.post("/api/deanonymize", response_model=DeanonymizeResponse)
async def deanonymize_text_endpoint(request: DeanonymizeRequest):
    if not isinstance(request.text, str) or not isinstance(request.entities, list):
        raise HTTPException(status_code=400, detail="Input 'text' must be a string and 'entities' must be a list.")

    original_text = anonymizer.deanonymize_text(request.text, request.entities)

    return {"original_text": original_text}


@app.post("/api/cohere-analyze", response_model=CohereAnalyzeResponse)
async def cohere_analyze_endpoint(request: AnonymizeRequest):
    if not isinstance(request.text, str):
        raise HTTPException(status_code=400, detail="Input 'text' must be a string.")

    analysis_text = await get_cohere_analysis(request.text)

    return {"analysis": analysis_text}
