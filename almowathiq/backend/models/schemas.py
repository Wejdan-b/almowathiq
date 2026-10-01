from pydantic import BaseModel, Field
from typing import List


class Source(BaseModel):
    scholar: str
    title: str
    url: str


class VerificationResponse(BaseModel):
    content_type: str = "fatwa"

    status: str

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )

    extracted_text: str

    correct_text: str

    missing_context: List[str]

    explanation: str

    source: Source