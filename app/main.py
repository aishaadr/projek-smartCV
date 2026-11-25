import io
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from app.schemas import AnalyzeResponse
from app.model import evaluate_cv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SmartCV - AI Resume Matcher",
    description="Microservice AI pemadanan CV menggunakan Hugging Face Transformers & KeyBERT",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "running", "service": "SmartCV HuggingFace Matcher"}

@app.get("/categories")
def get_categories():
    """Kategori posisi pekerjaan populer."""
    return {
        "categories": [
            "Software Engineer",
            "Data Scientist",
            "Machine Learning Engineer",
            "DevOps Engineer",
            "Frontend Developer",
            "Backend Developer",
            "UI/UX Designer",
            "Product Manager",
            "QA Engineer"
        ]
    }

@app.post("/analyze-cv", response_model=AnalyzeResponse)
async def analyze_cv(
    file: UploadFile = File(...),
    target_position: str = Form(...),
    job_description: str = Form(...)
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Hanya file resume berekstensi PDF yang didukung.")
    try:
        contents = await file.read()
        pdf_file = io.BytesIO(contents)
        
        # Ekstrak teks dari PDF
        reader = PdfReader(pdf_file)
        cv_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                cv_text += text + "\n"
        
        if not cv_text.strip():
            raise HTTPException(status_code=400, detail="PDF kosong atau berupa hasil scan gambar tanpa teks.")
        
        # Evaluasi menggunakan Model Hugging Face
        score, matched, missing, breakdown, recs = evaluate_cv(
            cv_text, job_description, target_position
        )
        return AnalyzeResponse(
            match_score=score,
            skills_found=matched,
            skills_missing=missing,
            breakdown=breakdown,
            recommendations=recs,
            ml_confidence=float(score),
            gemini_score=score
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error analisa CV: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan pemrosesan AI: {str(e)}")

