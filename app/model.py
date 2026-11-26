import os
import re
from typing import List, Tuple, Dict, Any, Optional

# Prevent tokenizer multithreading deadlocks on macOS/Linux
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT

_MODEL_CACHE: Optional[Tuple[SentenceTransformer, KeyBERT]] = None


def get_models() -> Tuple[SentenceTransformer, KeyBERT]:
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
        kw_model = KeyBERT(model=embedder)
        _MODEL_CACHE = (embedder, kw_model)
    return _MODEL_CACHE


# Industry skill taxonomy covering Data/AI, Web/RPL, IT Support/TKJ, UI/UX-DKV, and Admin Data
TECH_SKILLS_TAXONOMY = [
    # AI, ML & Data Science
    "Machine Learning", "Deep Learning", "Natural Language Processing", "NLP",
    "Computer Vision", "Generative AI", "Large Language Models", "LLM",
    "Supervised Learning", "Unsupervised Learning", "Reinforcement Learning",
    "Time Series", "Classification", "Regression", "Clustering",
    "Feature Engineering", "Class Imbalance", "Data Leakage", "Cross-Validation",
    "Stratified Cross-Validation", "Exploratory Data Analysis", "EDA",
    "Sentiment Analysis", "Customer Segmentation", "Model Deployment",
    "TensorFlow", "PyTorch", "Keras", "Scikit-Learn", "Pandas", "NumPy",
    "SciPy", "OpenCV", "Hugging Face", "Transformers", "Streamlit",

    # Web Development / RPL
    "HTML", "CSS", "JavaScript", "TypeScript", "PHP", "Laravel", "CodeIgniter",
    "React", "Vue", "Next.js", "Node.js", "Tailwind CSS", "Bootstrap",
    "Python", "R", "SQL", "PostgreSQL", "MySQL", "MongoDB",
    "REST API", "Git", "GitHub", "Docker", "FastAPI", "Flask", "Django",

    # IT Support / TKJ & Networking
    "Networking", "TCP/IP", "MikroTik", "Cisco", "LAN/WAN", "Routing", "Switching",
    "Hardware Troubleshooting", "Software Troubleshooting", "Linux",
    "Windows Server", "Active Directory", "Helpdesk", "Ticketing System",
    "IT Support", "Network Security", "Firewall", "Crimping",

    # UI/UX & DKV
    "Figma", "Adobe Photoshop", "Adobe Illustrator", "Canva", "CorelDraw",
    "Wireframing", "Prototyping", "UI Design", "UX Research", "Design System",
    "User Flow", "Usability Testing", "Typography", "Graphic Design",

    # Admin Data & Analytics Tools
    "Microsoft Excel", "Microsoft Word", "Google Sheets", "Data Entry",
    "Data Cleaning", "Pivot Table", "VLOOKUP", "Tableau", "Power BI",
    "Google Data Studio", "Database Systems", "Data Mining", "Reporting",

    # Soft Skills & Business
    "Critical Thinking", "Problem Solving", "Communication Skills",
    "Collaboration", "Business Analysis", "Leadership", "Time Management",
    "Teamwork", "Data Structures & Algorithms",
]

SYNONYMS_MAPPING = {
    "nlp": ["nlp", "natural language processing", "pengolahan teks", "pemrosesan bahasa alami", "text processing"],
    "natural language processing": ["natural language processing", "nlp", "pemrosesan bahasa alami", "pengolahan teks"],
    "machine learning": ["machine learning", "ml", "pembelajaran mesin"],
    "deep learning": ["deep learning", "dl", "pembelajaran mendalam", "neural networks"],
    "class imbalance": ["class imbalance", "imbalanced dataset", "ketimpangan kelas", "imbalance", "imbalanced records"],
    "data leakage": ["data leakage", "leakage prevention", "kebocoran data"],
    "feature engineering": ["feature engineering", "rekayasa fitur"],
    "cross-validation": ["cross-validation", "cross validation", "validasi silang", "stratified cross-validation"],
    "stratified cross-validation": ["stratified cross-validation", "stratified cross validation", "validasi silang"],
    "exploratory data analysis": ["exploratory data analysis", "eda"],
    "eda": ["eda", "exploratory data analysis", "analisis data eksploratif"],
    "time series": ["time series", "deret waktu"],
    "sentiment analysis": ["sentiment analysis", "analisis sentimen", "klasifikasi sentimen", "sentiment classifier"],
    "customer segmentation": ["customer segmentation", "segmentasi pelanggan", "klasterisasi"],
    "clustering": ["clustering", "klasterisasi", "cluster"],
    "critical thinking": ["critical thinking", "berpikir kritis"],
    "problem solving": ["problem solving", "pemecahan masalah"],
    "communication skills": ["communication skills", "communication", "komunikasi"],
    "collaboration": ["collaboration", "kerja sama tim", "teamwork"],
    "teamwork": ["teamwork", "kerja sama tim", "collaboration"],
    "business analysis": ["business analysis", "analisis bisnis", "business analyst", "analisis kebutuhan"],
    "unsupervised learning": ["unsupervised learning", "unsupervised machine learning"],
    "supervised learning": ["supervised learning", "supervised machine learning"],
    "model deployment": ["model deployment", "deployment", "deployed", "production-ready"],
    "microsoft excel": ["microsoft excel", "excel", "ms excel"],
    "microsoft word": ["microsoft word", "ms word"],
    "adobe photoshop": ["adobe photoshop", "photoshop"],
    "adobe illustrator": ["adobe illustrator", "illustrator"],
    "ui design": ["ui design", "user interface", "desain antarmuka"],
    "ux research": ["ux research", "user research", "riset pengguna"],
    "hardware troubleshooting": ["hardware troubleshooting", "perbaikan hardware", "troubleshooting perangkat keras"],
    "software troubleshooting": ["software troubleshooting", "instalasi os", "troubleshooting software"],
    "networking": ["networking", "jaringan komputer", "instalasi jaringan"],
    "data entry": ["data entry", "input data", "entri data"],
    "react": ["react", "react.js", "reactjs"],
    "vue": ["vue", "vue.js", "vuejs"],
    "next.js": ["next.js", "nextjs"],
    "node.js": ["node.js", "nodejs"],
}

INDONESIAN_STOPWORDS = {
    "yang", "untuk", "pada", "ke", "para", "namun", "menurut", "antara", "dia", "dua",
    "ia", "seperti", "jika", "sehingga", "kembali", "dan", "ini", "karena", "oleh",
    "saat", "harus", "sementara", "setelah", "belum", "kami", "kamu", "anda", "mereka",
    "kita", "akan", "bisa", "ada", "tidak", "dapat", "dari", "dalam", "dengan",
    "adalah", "sebagai", "serta", "secara", "menjadi", "memiliki", "terkait", "tentang",
    "posisi", "peran", "utama", "tanggung", "jawab", "kualifikasi", "dibutuhkan", "tersebut",
    "pengalaman", "proyek", "dataset", "tingkat", "bidang", "dunia", "nyata", "penuh",
    "siap", "kemampuan", "rekam", "jejak", "seluruh", "latar", "belakang", "terbukti",
}


def clean_kerning(text: str) -> str:
    if not text:
        return ""

    known_kerning = [
        (r"Tensor\s*fl\s*o\s*w", "TensorFlow"),
        (r"Pytorc\s*h", "PyTorch"),
        (r"Pytho\s*n", "Python"),
        (r"S\s*Q\s*L", "SQL"),
        (r"Microsoft\s*Exce\s*l", "Microsoft Excel"),
        (r"Pro\s*j\s*ect", "Project"),
        (r"D\s*ocuments", "Documents"),
        (r"Linkedl\s*n|Linked\s*In", "LinkedIn"),
        (r"S\s*treamlit", "Streamlit"),
        (r"K\s*eras", "Keras"),
        (r"Java\s*S\s*cript", "JavaScript"),
        (r"Type\s*S\s*cript", "TypeScript"),
        (r"Photo\s*shop", "Photoshop"),
    ]
    for pattern, repl in known_kerning:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_skills_hybrid(
    job_description: str,
    kw_model: Optional[KeyBERT] = None,
) -> List[str]:
    if kw_model is None:
        _, kw_model = get_models()

    clean_jd = clean_kerning(job_description)
    detected_skills: List[str] = []

    for skill in TECH_SKILLS_TAXONOMY:
        skill_lower = skill.lower()
        terms = [skill_lower]
        if skill_lower in SYNONYMS_MAPPING:
            terms.extend(SYNONYMS_MAPPING[skill_lower])

        for term in terms:
            pattern = r"\b" + re.escape(term) + r"\b"
            if re.search(pattern, clean_jd, re.IGNORECASE):
                if skill not in detected_skills:
                    detected_skills.append(skill)
                break

    try:
        raw_keywords = kw_model.extract_keywords(
            clean_jd,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=15,
        )
        for kw, _ in raw_keywords:
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


def evaluate_cv(
    cv_text: str,
    job_description: str,
    target_position: str,
    models: Optional[Tuple[SentenceTransformer, KeyBERT]] = None,
) -> Tuple[int, List[str], List[str], Dict[str, int], List[str], Dict[str, Any]]:
    embedder, kw_model = models if models is not None else get_models()

    clean_cv = clean_kerning(cv_text)
    clean_jd = clean_kerning(job_description)

    target_skills = extract_skills_hybrid(clean_jd, kw_model=kw_model)
    if not target_skills:
        target_skills = ["Python", "Problem Solving", "Communication Skills", "SQL"]

    matched_skills: List[str] = []
    missing_skills: List[str] = []

    cv_lower = clean_cv.lower()
    # Encode CV snippet once outside the skill loop to avoid redundant CPU work
    cv_sample_emb = embedder.encode(clean_cv[:2500], convert_to_tensor=True, device="cpu")

    for skill in target_skills:
        skill_lower = skill.lower()
        terms = [skill_lower]
        if skill_lower in SYNONYMS_MAPPING:
            terms.extend(SYNONYMS_MAPPING[skill_lower])

        found = False
        for term in terms:
            pattern = r"\b" + re.escape(term) + r"\b"
            if re.search(pattern, cv_lower):
                found = True
                break

        if found:
            matched_skills.append(skill)
        else:
            try:
                skill_emb = embedder.encode(skill, convert_to_tensor=True, device="cpu")
                sim = float(util.cos_sim(skill_emb, cv_sample_emb)[0][0])
                if sim > 0.48:
                    matched_skills.append(skill)
                else:
                    missing_skills.append(skill)
            except Exception:
                missing_skills.append(skill)

    # Identify extra skills present in CV beyond the core JD requirements
    all_cv_skills: List[str] = []
    for skill in TECH_SKILLS_TAXONOMY:
        skill_lower = skill.lower()
        terms = [skill_lower]
        if skill_lower in SYNONYMS_MAPPING:
            terms.extend(SYNONYMS_MAPPING[skill_lower])
        for term in terms:
            if re.search(r"\b" + re.escape(term) + r"\b", cv_lower):
                if skill not in all_cv_skills:
                    all_cv_skills.append(skill)
                break

    surplus_skills = [s for s in all_cv_skills if s not in matched_skills]

    # Full-text semantic similarity
    try:
        cv_emb = embedder.encode(clean_cv[:3500], convert_to_tensor=True, device="cpu")
        jd_emb = embedder.encode(clean_jd[:3500], convert_to_tensor=True, device="cpu")
        cosine_sim = float(util.cos_sim(cv_emb, jd_emb)[0][0])
    except Exception:
        cosine_sim = 0.50

    normalized_sim = max(0.0, min(1.0, (cosine_sim - 0.20) / 0.55))
    total_req = len(matched_skills) + len(missing_skills)
    skill_ratio = (len(matched_skills) / total_req) if total_req > 0 else 0.5

    weighted_score = (skill_ratio * 0.55) + (normalized_sim * 0.45)
    final_score = int(round(weighted_score * 100))
    final_score = max(15, min(98, final_score))

    tech_score = int(round(skill_ratio * 100))
    exp_score = int(round(normalized_sim * 100))
    ind_score = int(round((tech_score * 0.6) + (exp_score * 0.4)))

    breakdown = {
        "technical": max(10, min(100, tech_score)),
        "experience": max(10, min(100, exp_score)),
        "industry": max(10, min(100, ind_score)),
    }

    recommendations: List[str] = []
    if missing_skills:
        top_missing = ", ".join(missing_skills[:3])
        recommendations.append(
            f"Keterampilan Prioritas: Cantumkan contoh proyek nyata atau kualifikasi yang menonjolkan **{top_missing}**."
        )
        recommendations.append(
            f"Penguatan Profil: Lengkapi portofolio dengan studi kasus terkait **{missing_skills[0]}** untuk meningkatkan daya saing."
        )
    else:
        recommendations.append(
            "Kesesuaian Sempurna: Seluruh kompetensi teknis dan spesifikasi kualifikasi yang dicari telah terpenuhi dengan sangat baik pada profil Anda!"
        )

    recommendations.append(
        f"Strategi Ringkasan: Pastikan kata kunci teknis posisi **{target_position}** termuat jelas pada ringkasan profil di bagian atas resume."
    )
    recommendations.append(
        "Kuantifikasi Pencapaian: Pertahankan format penyajian metrik performa atau hasil kerja terukur pada riwayat proyek."
    )

    recruitment_metrics = {
        "cosine_similarity": round(cosine_sim * 100, 1),
        "matched_count": len(matched_skills),
        "total_required": total_req,
        "surplus_count": len(surplus_skills),
        "surplus_skills": surplus_skills,
    }

    return final_score, matched_skills, missing_skills, breakdown, recommendations, recruitment_metrics
