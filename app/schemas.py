from pydantic import BaseModel
from typing import List, Dict

class AnalyzeResponse(BaseModel):
    match_score: int
    skills_found: List[str]
    skills_missing: List[str]
    breakdown: Dict[str, int]
    recommendations: List[str]
    ml_confidence: float
    gemini_score: int

