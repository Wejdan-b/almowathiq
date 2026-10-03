from pydantic import BaseModel, Field
from typing import List, Optional


class Source(BaseModel):
    scholar: str = ""
    title: str = ""
    url: str = ""


class VerificationResponse(BaseModel):
    content_type: str = "fatwa"

    status: str

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )

    extracted_text: str

    correct_text: Optional[str] = None

    missing_context: List[str] = Field(
        default_factory=list
    )

    explanation: str

    source: Optional[Source] = None 