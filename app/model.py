import os
import re
from typing import List, Tuple, Dict

# Cegah konflik multithreading tokenizers pada macOS/Linux
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT

# Load pre-trained models dari Hugging Face dengan device='cpu' untuk stabilitas maksimal di macOS/Windows/Linux
print("Loading Hugging Face models on CPU...")
embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
kw_model = KeyBERT(model=embedder)
print("Hugging Face models ready!")

def clean_text(text: str) -> str:
    """Membersihkan teks dari karakter aneh dan spasi berlebih."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_skills_keybert(job_description: str) -> List[str]:
    """
    Mengekstrak kata kunci keahlian secara otomatis dari Job Description
    menggunakan KeyBERT (Tanpa kamus manual).
    """
    keywords = kw_model.extract_keywords(
        job_description, 
        keyphrase_ngram_range=(1, 2), 
        stop_words='english', 
        top_n=12
    )
    return [kw[0].title() for kw in keywords]

def evaluate_cv(cv_text: str, job_description: str, target_position: str) -> Tuple[int, List[str], List[str], Dict[str, int], List[str]]:
    """
    Menganalisis kecocokan CV terhadap Job Description menggunakan Semantic Embedding.
    """
    clean_cv = clean_text(cv_text)
    clean_jd = clean_text(job_description)
    
    # 1. Hitung Semantic Cosine Similarity antara CV dan Job Description
    cv_emb = embedder.encode(clean_cv, convert_to_tensor=True, device='cpu')
    jd_emb = embedder.encode(clean_jd, convert_to_tensor=True, device='cpu')
    cosine_sim = float(util.cos_sim(cv_emb, jd_emb)[0][0])
    
    # 2. Ekstrak Keterampilan dari Job Description
    target_skills = extract_skills_keybert(job_description)
    
    matched_skills = []
    missing_skills = []
    
    # 3. Cek keberadaan skill di CV secara semantik / leksikal
    for skill in target_skills:
        skill_pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(skill_pattern, clean_cv.lower()):
            matched_skills.append(skill)
        else:
            # Semantic check: jika embedding skill sangat dekat dengan kalimat di CV
            skill_emb = embedder.encode(skill, convert_to_tensor=True, device='cpu')
            skill_sim = float(util.cos_sim(skill_emb, cv_emb)[0][0])
            if skill_sim > 0.45:
                matched_skills.append(skill)
            else:
                missing_skills.append(skill)
                
    # 4. Hitung Rasio Keterampilan
    total_skills = len(matched_skills) + len(missing_skills)
    skill_ratio = (len(matched_skills) / total_skills) if total_skills > 0 else 0.5
    
    # 5. Skor Akhir (Gabungan 60% Semantic Similarity + 40% Skill Match Ratio)
    normalized_sim = max(0.0, min(1.0, (cosine_sim - 0.2) / 0.6))
    final_score = int(round(((normalized_sim * 0.6) + (skill_ratio * 0.4)) * 100))
    final_score = max(5, min(98, final_score))
    
    # 6. Hitung Breakdown Kategori (Technical, Experience, Industry)
    tech_score = int(round(skill_ratio * 100))
    exp_score = int(round(normalized_sim * 100))
    ind_score = int(round(((tech_score + exp_score) / 2)))
    breakdown = {
        "technical": max(10, min(100, tech_score)),
        "experience": max(10, min(100, exp_score)),
        "industry": max(10, min(100, ind_score))
    }
    
    # 7. Generate Rekomendasi Pintar Dinamis Berdasarkan Missing Skills
    recommendations = []
    if missing_skills:
        top_missing = ", ".join(missing_skills[:3])
        recommendations.append(f"Tambahkan pengalaman proyek atau studi kasus yang menggunakan keahlian: {top_missing}.")
        recommendations.append(f"Cantumkan sertifikasi atau pelatihan yang relevan dengan {missing_skills[0]} untuk memperkuat kualifikasi.")
    else:
        recommendations.append("Keahlian teknis kamu sudah sangat cocok dengan kualifikasi pekerjaan ini!")
    
    recommendations.append(f"Gunakan kata kunci dan istilah teknis standar industri untuk posisi '{target_position}' pada ringkasan profilmu.")
    return final_score, matched_skills, missing_skills, breakdown, recommendations
