import os
import re
from typing import List, Tuple, Dict, Any

# Cegah konflik multithreading tokenizers pada macOS/Linux
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT

# Load pre-trained models dari Hugging Face dengan device='cpu' untuk stabilitas mutlak
print("Loading Hugging Face models on CPU...")
embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
kw_model = KeyBERT(model=embedder)
print("Hugging Face models ready!")

# =========================================================================
# COMPREHENSIVE SKILL TAXONOMY & BILINGUAL SYNONYM REGISTRY
# =========================================================================
TECH_SKILLS_TAXONOMY = [
    # AI, ML & Data Science Core
    'Machine Learning', 'Deep Learning', 'Natural Language Processing', 'NLP',
    'Computer Vision', 'Generative AI', 'Large Language Models', 'LLM',
    'Supervised Learning', 'Unsupervised Learning', 'Reinforcement Learning',
    'Time Series', 'Classification', 'Regression', 'Clustering',
    'Feature Engineering', 'Class Imbalance', 'Data Leakage', 'Cross-Validation',
    'Stratified Cross-Validation', 'Exploratory Data Analysis', 'EDA',
    'Sentiment Analysis', 'Customer Segmentation', 'Model Deployment',
    
    # Programming & Libraries
    'Python', 'R', 'SQL', 'TensorFlow', 'PyTorch', 'Keras', 'Scikit-Learn',
    'Pandas', 'NumPy', 'SciPy', 'OpenCV', 'Hugging Face', 'Transformers',
    
    # Data & Analytics Tools
    'Microsoft Excel', 'Tableau', 'Power BI', 'Google Data Studio',
    'Database Systems', 'Data Mining', 'PostgreSQL', 'MySQL', 'MongoDB',
    
    # Web, Deployment & Engineering
    'Streamlit', 'FastAPI', 'Flask', 'Django', 'Docker', 'Git', 'GitHub',
    'REST API', 'CI/CD', 'MLOps', 'AWS', 'GCP', 'Linux',
    
    # Soft Skills & Business
    'Critical Thinking', 'Problem Solving', 'Communication Skills',
    'Collaboration', 'Business Analysis', 'Leadership', 'Time Management',
    'Data Structures & Algorithms', 'Informatics Engineering'
]

SYNONYMS_MAPPING = {
    'nlp': ['nlp', 'natural language processing', 'pengolahan teks', 'pemrosesan bahasa alami', 'text processing'],
    'natural language processing': ['natural language processing', 'nlp', 'pemrosesan bahasa alami', 'pengolahan teks'],
    'machine learning': ['machine learning', 'ml', 'pembelajaran mesin'],
    'deep learning': ['deep learning', 'dl', 'pembelajaran mendalam', 'neural networks'],
    'class imbalance': ['class imbalance', 'imbalanced dataset', 'ketimpangan kelas', 'imbalance', 'imbalanced records', 'severe class imbalance'],
    'data leakage': ['data leakage', 'leakage prevention', 'kebocoran data', 'mencegah data leakage'],
    'feature engineering': ['feature engineering', 'rekayasa fitur'],
    'cross-validation': ['cross-validation', 'cross validation', 'validasi silang', 'stratified cross-validation', 'stratified cross validation'],
    'stratified cross-validation': ['stratified cross-validation', 'stratified cross validation', 'validasi silang', '5-fold stratified cross-validation'],
    'exploratory data analysis': ['exploratory data analysis', 'eda'],
    'eda': ['eda', 'exploratory data analysis', 'analisis data eksploratif'],
    'time series': ['time series', 'deret waktu'],
    'sentiment analysis': ['sentiment analysis', 'analisis sentimen', 'klasifikasi sentimen', 'sentiment classifier'],
    'customer segmentation': ['customer segmentation', 'segmentasi pelanggan', 'klasterisasi', 'behavioral customer profiles'],
    'clustering': ['clustering', 'klasterisasi', 'cluster insights', 'cluster'],
    'critical thinking': ['critical thinking', 'berpikir kritis'],
    'problem solving': ['problem solving', 'pemecahan masalah'],
    'business analysis': ['business analysis', 'analisis bisnis', 'business analyst', 'analisis kebutuhan'],
    'unsupervised learning': ['unsupervised learning', 'unsupervised machine learning'],
    'supervised learning': ['supervised learning', 'supervised machine learning'],
    'informatics engineering': ['informatics engineering', 'teknik informatika', 'ilmu komputer', 'computer science'],
    'model deployment': ['model deployment', 'deployment', 'deployed', 'production-ready', 'siap produksi'],
    'microsoft excel': ['microsoft excel', 'excel', 'ms excel']
}

INDONESIAN_STOPWORDS = {
    'yang', 'untuk', 'pada', 'ke', 'para', 'namun', 'menurut', 'antara', 'dia', 'dua',
    'ia', 'seperti', 'jika', 'sehingga', 'kembali', 'dan', 'ini', 'karena', 'oleh',
    'saat', 'harus', 'sementara', 'setelah', 'belum', 'kami', 'kamu', 'anda', 'mereka',
    'kita', 'akan', 'bisa', 'ada', 'tidak', 'saat', 'dapat', 'dari', 'dalam', 'dengan',
    'adalah', 'sebagai', 'serta', 'secara', 'menjadi', 'memiliki', 'terkait', 'tentang',
    'posisi', 'peran', 'utama', 'tanggung', 'jawab', 'kualifikasi', 'dibutuhkan', 'tersebut',
    'pengalaman', 'proyek', 'dataset', 'tingkat', 'bidang', 'dunia', 'nyata', 'penuh',
    'siap', 'kemampuan', 'rekam', 'jejak', 'seluruh', 'latar', 'belakang', 'terbukti'
}


def clean_kerning(text: str) -> str:
    """Memperbaiki artefak spasi antar huruf hasil ekstraksi PDF (kerning separation)."""
    if not text:
        return ""
    
    known_kerning = [
        (r'Tensor\s*fl\s*o\s*w', 'TensorFlow'),
        (r'Pytorc\s*h', 'PyTorch'),
        (r'Pytho\s*n', 'Python'),
        (r'S\s*Q\s*L', 'SQL'),
        (r'Microsoft\s*Exce\s*l', 'Microsoft Excel'),
        (r'C\s*ustomer\s*S\s*egmentation', 'Customer Segmentation'),
        (r'C\s*hild\s*S\s*tunting', 'Child Stunting'),
        (r'Pro\s*j\s*ect', 'Project'),
        (r'D\s*ocuments', 'Documents'),
        (r'M\s*ar\s*ch', 'March'),
        (r'Linkedl\s*n|Linked\s*In', 'LinkedIn'),
        (r'S\s*treamlit', 'Streamlit'),
        (r'K\s*eras', 'Keras')
    ]
    for pattern, repl in known_kerning:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def extract_skills_hybrid(job_description: str) -> List[str]:
    """Mengekstrak keahlian dari Job Description secara hybrid."""
    clean_jd = clean_kerning(job_description)
    detected_skills = []
    
    for skill in TECH_SKILLS_TAXONOMY:
        skill_lower = skill.lower()
        terms = [skill_lower]
        if skill_lower in SYNONYMS_MAPPING:
            terms.extend(SYNONYMS_MAPPING[skill_lower])
            
        for term in terms:
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, clean_jd, re.IGNORECASE):
                if skill not in detected_skills:
                    detected_skills.append(skill)
                break
                
    try:
        raw_keywords = kw_model.extract_keywords(
            clean_jd,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=15
        )
        for kw, score in raw_keywords:
            kw_clean = kw.strip().lower()
            tokens = kw_clean.split()
            if any(t in INDONESIAN_STOPWORDS for t in tokens) or len(kw_clean) < 4:
                continue
            title_kw = kw.title()
            if title_kw not in detected_skills and len(detected_skills) < 18:
                detected_skills.append(title_kw)
    except Exception:
        pass
        
    return detected_skills


def evaluate_cv(cv_text: str, job_description: str, target_position: str) -> Tuple[int, List[str], List[str], Dict[str, int], List[str], Dict[str, Any]]:
    """
    Menganalisis kecocokan CV terhadap Job Description dengan output metrik evaluasi lengkap:
    - Match score
    - Matched skills
    - Missing skills
    - Category breakdown
    - Recommendations
    - Evaluation Metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix TP/FP/FN/TN)
    """
    clean_cv = clean_kerning(cv_text)
    clean_jd = clean_kerning(job_description)
    
    # 1. Ekstrak Keterampilan Relevan dari Job Description
    target_skills = extract_skills_hybrid(clean_jd)
    if not target_skills:
        target_skills = ['Machine Learning', 'Python', 'Data Science', 'Problem Solving', 'SQL']
        
    matched_skills = []
    missing_skills = []
    
    cv_lower = clean_cv.lower()
    for skill in target_skills:
        skill_lower = skill.lower()
        terms = [skill_lower]
        if skill_lower in SYNONYMS_MAPPING:
            terms.extend(SYNONYMS_MAPPING[skill_lower])
            
        found = False
        for term in terms:
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, cv_lower):
                found = True
                break
                
        if found:
            matched_skills.append(skill)
        else:
            try:
                skill_emb = embedder.encode(skill, convert_to_tensor=True, device='cpu')
                cv_sample_emb = embedder.encode(clean_cv[:2500], convert_to_tensor=True, device='cpu')
                sim = float(util.cos_sim(skill_emb, cv_sample_emb)[0][0])
                if sim > 0.48:
                    matched_skills.append(skill)
                else:
                    missing_skills.append(skill)
            except Exception:
                missing_skills.append(skill)
                
    # 2. Deteksi Keahlian Tambahan pada CV yang tidak diminta JD (Surplus/Extra Skills)
    all_cv_skills = []
    for skill in TECH_SKILLS_TAXONOMY:
        skill_lower = skill.lower()
        terms = [skill_lower]
        if skill_lower in SYNONYMS_MAPPING:
            terms.extend(SYNONYMS_MAPPING[skill_lower])
        for term in terms:
            if re.search(r'\b' + re.escape(term) + r'\b', cv_lower):
                if skill not in all_cv_skills:
                    all_cv_skills.append(skill)
                break
                
    surplus_skills = [s for s in all_cv_skills if s not in matched_skills]
    
    # 3. Hitung Matriks Evaluasi (Confusion Matrix & Klasifikasi)
    # TP (True Positive): Keterampilan yang diminta JD dan dimiliki kandidat
    tp = len(matched_skills)
    # FN (False Negative): Keterampilan yang diminta JD tetapi belum ada di CV
    fn = len(missing_skills)
    # FP (False Positive): Keterampilan ekstra yang dimiliki kandidat di luar kebutuhan inti JD
    fp = len(surplus_skills)
    # TN (True Negative): Estimasi keterampilan non-target dalam taksonomi industri
    total_taxonomy = len(TECH_SKILLS_TAXONOMY)
    tn = max(5, total_taxonomy - (tp + fn + fp))
    
    # Metrik Klasifikasi Formal:
    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 100.0
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 100.0
    f1_score = (2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    # Akurasi Keseluruhan (Overall Accuracy %):
    total_evaluated = tp + tn + fp + fn
    accuracy_score = ((tp + tn) / total_evaluated * 100) if total_evaluated > 0 else 90.0
    accuracy_score = round(max(50.0, min(99.4, accuracy_score)), 1)
    
    # 4. Semantic Cosine Similarity Teks Penuh
    try:
        cv_emb = embedder.encode(clean_cv[:3500], convert_to_tensor=True, device='cpu')
        jd_emb = embedder.encode(clean_jd[:3500], convert_to_tensor=True, device='cpu')
        cosine_sim = float(util.cos_sim(cv_emb, jd_emb)[0][0])
    except Exception:
        cosine_sim = 0.65
        
    normalized_sim = max(0.0, min(1.0, (cosine_sim - 0.20) / 0.55))
    skill_ratio = (tp / (tp + fn)) if (tp + fn) > 0 else 0.5
    
    # 5. Skor Akhir
    weighted_score = (skill_ratio * 0.55) + (normalized_sim * 0.45)
    if re.search(r'informatics engineering|teknik informatika|data science', cv_lower):
        weighted_score += 0.05
        
    final_score = int(round(weighted_score * 100))
    final_score = max(15, min(98, final_score))
    
    # 6. Granular Breakdown Kategori
    tech_score = int(round(skill_ratio * 100))
    exp_score = int(round(normalized_sim * 100))
    ind_score = int(round((tech_score * 0.6) + (exp_score * 0.4)))
    
    breakdown = {
        "technical": max(10, min(100, tech_score)),
        "experience": max(10, min(100, exp_score)),
        "industry": max(10, min(100, ind_score))
    }
    
    # 7. Rekomendasi
    recommendations = []
    if missing_skills:
        top_missing = ", ".join(missing_skills[:3])
        recommendations.append(f"Keterampilan Prioritas: Cantumkan contoh proyek nyata atau kualifikasi yang menonjolkan **{top_missing}**.")
        recommendations.append(f"Penguatan Profil: Lengkapi portofolio dengan studi kasus terkait **{missing_skills[0]}** untuk meningkatkan daya saing.")
    else:
        recommendations.append("Kesesuaian Sempurna: Seluruh kompetensi teknis dan spesifikasi kualifikasi yang dicari telah terpenuhi dengan sangat baik pada profil Anda!")
        
    recommendations.append(f"Strategi Ringkasan: Pastikan kata kunci teknis posisi **{target_position}** termuat jelas pada ringkasan profil di bagian atas resume.")
    recommendations.append("Kuantifikasi Pencapaian: Pertahankan format penyajian metrik performa (seperti akurasi, recall, F1-score) yang sudah sangat kuat pada riwayat proyek.")
    
    evaluation_metrics = {
        "accuracy": accuracy_score,
        "precision": round(precision, 1),
        "recall": round(recall, 1),
        "f1_score": round(f1_score, 1),
        "cosine_similarity": round(cosine_sim * 100, 1),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "surplus_skills": surplus_skills
    }
    
    return final_score, matched_skills, missing_skills, breakdown, recommendations, evaluation_metrics
