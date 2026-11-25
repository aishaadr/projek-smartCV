from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class EvaluationMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    cosine_similarity: float
    tp: int
    fn: int
    fp: int
    tn: int
    surplus_skills: Optional[List[str]] = []

class AnalyzeResponse(BaseModel):
    match_score: int
    skills_found: List[str]
    skills_missing: List[str]
    breakdown: Dict[str, int]
    recommendations: List[str]
    ml_confidence: float
    gemini_score: int
    evaluation_metrics: Optional[EvaluationMetrics] = None
