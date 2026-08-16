import os
from dataclasses import dataclass
from typing import List, Dict, Optional
from PIL import Image
import pytesseract
from rapidfuzz import fuzz

SCHEMA = {
    "invoice_number": ["invoice no", "invoice number", "inv no", "factura no"],
    "invoice_date": ["date", "invoice date", "fecha"],
    "total_amount": ["total", "amount due", "balance due", "total due", "importe total"],
}

CONFIDENCE_THRESHOLD = 60

@dataclass
class Token:
    text: str
    x: int
    y: int
    w: int
    h: int
    conf: float

class DocumentIntelligenceAgent:
    def __init__(self, confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold

    def extract_tokens(self, image: Image.Image) -> List[Token]:
        """Aplica OCR y extrae tokens con coordenadas espaciales y nivel de confianza."""
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        tokens = []
        n_boxes = len(data["text"])
        
        for i in range(n_boxes):
            text = data["text"][i].strip()
            conf = float(data["conf"][i])
            if text and conf >= self.confidence_threshold:
                tokens.append(Token(
                    text=text,
                    x=data["left"][i],
                    y=data["top"][i],
                    w=data["width"][i],
                    h=data["height"][i],
                    conf=conf
                ))
        return tokens

    def extract_fields_from_tokens(self, tokens: List[Token]) -> Dict[str, Dict[str, Any]]:
        """Extracción basada en proximidad espacial y correspondencia difusa con el esquema."""
        extracted = {k: {"value": None, "confidence": 0.0, "provenance": None} for k in SCHEMA}
        
        for idx, token in enumerate(tokens):
            for field, cues in SCHEMA.items():
                for cue in cues:
                    # Coincidencia difusa del token contra las palabras clave del esquema
                    similarity = fuzz.partial_ratio(cue.lower(), token.text.lower())
                    if similarity > 85:
                        # Recuperar los siguientes tokens adyacentes horizontalmente
                        candidate_values = []
                        for next_token in tokens[idx + 1: idx + 4]:
                            if abs(next_token.y - token.y) < 15:  # Misma línea visual
                                candidate_values.append(next_token)
                        
                        if candidate_values and not extracted[field]["value"]:
                            val_text = " ".join([t.text for t in candidate_values])
                            avg_conf = sum([t.conf for t in candidate_values]) / len(candidate_values)
                            extracted[field] = {
                                "value": val_text,
                                "confidence": round(avg_conf, 2),
                                "provenance": {
                                    "bbox": [token.x, token.y, candidate_values[-1].x + candidate_values[-1].w, candidate_values[-1].y + candidate_values[-1].h]
                                }
                            }
        return extracted