import io
import os
import re

# Cegah konflik tokenizers multithreading pada macOS/Linux
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import plotly.graph_objects as go
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT

# ==========================================
# PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="SmartCV Analytics — AI Resume & Job Matcher",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Clean & Human-Crafted CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Top Banner Header */
    .hero-container {
        padding: 1.5rem 0 1rem 0;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 1.8rem;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.25rem;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 0;
    }
    
    /* KPI Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: transform 0.15s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.08);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
    }
    
    /* Skill Badges */
    .badge-matched {
        display: inline-block;
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.2rem 0.25rem;
    }
    .badge-missing {
        display: inline-block;
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.2rem 0.25rem;
    }
    
    /* Recommendations Box */
    .recom-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 1rem 1.25rem;
        border-radius: 0 10px 10px 0;
        margin-bottom: 0.75rem;
        font-size: 0.95rem;
        color: #334155;
    }
    
    /* Subtle divider */
    .custom-divider {
        height: 1px;
        background: #E2E8F0;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# MODEL INITIALIZATION (CACHED)
# ==========================================
@st.cache_resource(show_spinner="Memuat model Hugging Face (Sentence-Transformers & KeyBERT)...")
def load_ai_models():
    """Memuat model NLP ke memori satu kali saja dengan device='cpu' untuk stabilitas mutlak."""
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    kw_model = KeyBERT(model=embedder)
    return embedder, kw_model

embedder, kw_model = load_ai_models()


# ==========================================
# CORE NLP & RESUME MATCHING ENGINE
# ==========================================
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_skills_keybert(job_description: str, top_n: int = 12):
    """Mengekstrak kata kunci keterampilan secara semantik dari job description."""
    keywords = kw_model.extract_keywords(
        job_description,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=top_n
    )
    return [kw[0].title() for kw in keywords]

def evaluate_cv(cv_text: str, job_description: str, target_position: str):
    clean_cv = clean_text(cv_text)
    clean_jd = clean_text(job_description)
    
    # 1. Hitung Semantic Cosine Similarity (Sentence Transformers)
    cv_emb = embedder.encode(clean_cv, convert_to_tensor=True, device='cpu')
    jd_emb = embedder.encode(clean_jd, convert_to_tensor=True, device='cpu')
    cosine_sim = float(util.cos_sim(cv_emb, jd_emb)[0][0])
    
    # 2. Ekstrak Keterampilan Kunci dari Job Description
    target_skills = extract_skills_keybert(job_description)
    
    matched_skills = []
    missing_skills = []
    
    # 3. Evaluasi Semantik & Leksikal Keterampilan pada CV
    for skill in target_skills:
        skill_pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(skill_pattern, clean_cv.lower()):
            matched_skills.append(skill)
        else:
            skill_emb = embedder.encode(skill, convert_to_tensor=True, device='cpu')
            skill_sim = float(util.cos_sim(skill_emb, cv_emb)[0][0])
            if skill_sim > 0.45:
                matched_skills.append(skill)
            else:
                missing_skills.append(skill)
                
    # 4. Rasio Kecocokan Keterampilan
    total_skills = len(matched_skills) + len(missing_skills)
    skill_ratio = (len(matched_skills) / total_skills) if total_skills > 0 else 0.5
    
    # 5. Normalisasi Skor Akhir (60% Semantik Similarity + 40% Keterampilan)
    normalized_sim = max(0.0, min(1.0, (cosine_sim - 0.2) / 0.6))
    final_score = int(round(((normalized_sim * 0.6) + (skill_ratio * 0.4)) * 100))
    final_score = max(8, min(97, final_score))
    
    # 6. Breakdown Kategori
    tech_score = int(round(skill_ratio * 100))
    exp_score = int(round(normalized_sim * 100))
    ind_score = int(round(((tech_score + exp_score) / 2)))
    
    breakdown = {
        "Technical Skills Match": max(10, min(100, tech_score)),
        "Experience & Context Relevance": max(10, min(100, exp_score)),
        "Industry Terminology Alignment": max(10, min(100, ind_score))
    }
    
    # 7. Rekomendasi Karir & Penulisan CV
    recommendations = []
    if missing_skills:
        top_missing = ", ".join(missing_skills[:3])
        recommendations.append(f"Keahlian Prioritas: Pertimbangkan untuk menambahkan pengalaman riil atau portfolio proyek yang mencakup **{top_missing}**.")
        recommendations.append(f"Penguatan Kualifikasi: Pelajari atau lampirkan sertifikasi terkait **{missing_skills[0]}** guna meningkatkan daya saing profil.")
    else:
        recommendations.append("Kesesuaian Kompetensi Unggul: Seluruh keterampilan kunci yang dibutuhkan dalam deskripsi lowongan telah teridentifikasi pada profil Anda.")
        
    recommendations.append(f"Optimalisasi Ringkasan Profil: Masukkan kata kunci teknis standar industri untuk posisi **{target_position}** pada bagian rangkuman pengalaman di awal CV.")
    recommendations.append("Metrik Pencapaian Kuantitatif: Gunakan formula 'Accomplished [X] as measured by [Y], by doing [Z]' pada riwayat pekerjaan untuk memperjelas dampak kontribusi Anda.")
    
    return final_score, matched_skills, missing_skills, breakdown, recommendations


# ==========================================
# PRESET JOB SAMPLES FOR EASY TESTING
# ==========================================
PRESET_JOBS = {
    "Pilih Template Lowongan (Opsional)": "",
    "Data Scientist": """Kami mencari Data Scientist yang memiliki pemahaman kuat dalam Python, Machine Learning, Scikit-Learn, dan Pandas.
Tanggung jawab meliputi eksplorasi data (EDA), pengembangan model prediktif, evaluasi performa model, serta komunikasi temuan bisnis melalui visualisasi data interaktif.
Kualifikasi:
- Menguasai Python, SQL, dan manipulasi data besar.
- Berpengalaman dengan algoritma Machine Learning (Regression, Classification, Clustering).
- Pemahaman tentang Natural Language Processing (NLP) atau Deep Learning merupakan nilai tambah.
- Mampu menyajikan insight teknis kepada pemangku kepentingan non-teknis.""",

    "Machine Learning Engineer": """Dicari Machine Learning Engineer untuk mendesain, melatih, dan men-deploy model AI ke lingkungan produksi.
Keahlian teknis yang dibutuhkan:
- Python, PyTorch atau TensorFlow, Docker, dan CI/CD pipeline.
- Pengalaman membangun REST API menggunakan FastAPI atau Flask untuk inference model.
- Memahami konsep MLOps, model monitoring, dan data drift.
- Pengalaman dengan Hugging Face Transformers dan arsitektur LLM.""",

    "Full-Stack Web Developer": """Mencari Full-Stack Developer dengan fokus pada arsitektur modern web application.
Kualifikasi:
- Mahir dalam JavaScript/TypeScript, React.js atau Vue.js untuk sisi frontend.
- Menguasai Node.js, Python, atau PHP (Laravel) untuk sisi backend dan RESTful API.
- Berpengalaman dalam desain database relasional (PostgreSQL/MySQL) dan query optimization.
- Memahami version control Git, unit testing, dan cloud deployment.""",

    "UI/UX & Product Designer": """Kami membutuhkan UI/UX Designer yang berfokus pada user-centered design, pembuatan prototype interaktif, dan usability testing.
Kualifikasi utama:
- Sangat mahir menggunakan Figma untuk wireframing, high-fidelity design, dan design system.
- Memiliki portfolio desain web dan mobile yang mengedepankan kemudahan navigasi dan estetika visual.
- Mampu melakukan user research, journey mapping, dan interview pengguna.
- Berkolaborasi erat dengan tim software engineering untuk memastikan kepatuhan implementasi desain."""
}


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1586281380349-632531db7ed4?w=400&auto=format&fit=crop&q=80", use_container_width=True)
    st.markdown("### ⚙️ Konfigurasi Lowongan")
    
    selected_preset = st.selectbox(
        "Gunakan Contoh Posisi Pekerjaan:",
        options=list(PRESET_JOBS.keys())
    )
    
    st.markdown("---")
    st.markdown("### 🧠 Arsitektur Model")
    st.markdown("""
    - **Embedder**: `all-MiniLM-L6-v2`  
      *(Hugging Face Sentence-Transformers)*
    - **Skill Extractor**: `KeyBERT`  
      *(Unsupervised Semantic Keyword Extraction)*
    - **Similarity Metric**: Cosine Similarity
    """)
    
    st.markdown("---")
    st.markdown("### 👩‍💻 Dibuat Oleh")
    st.markdown("""
    **Aisha Dwi Rahmawati**  
    Repository: [github.com/aishaadr/projek-smartCV](https://github.com/aishaadr/projek-smartCV)
    """)


# ==========================================
# MAIN APPLICATION INTERFACE
# ==========================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">SmartCV Analytics</div>
    <p class="hero-subtitle">Platform analitik pencocokan resume terhadap kualifikasi pekerjaan berbasis Semantic NLP & Transformative AI.</p>
</div>
""", unsafe_allow_html=True)

col_input_left, col_input_right = st.columns([1, 1], gap="large")

with col_input_left:
    st.subheader("1. Unggah Berkas Resume (CV)")
    uploaded_file = st.file_uploader(
        "Pilih file resume dalam format PDF:",
        type=["pdf"],
        help="Pastikan PDF memuat teks yang dapat dibaca (bukan hasil scan gambar mati)."
    )
    
    target_pos = st.text_input(
        "Posisi Pekerjaan yang Dituju:",
        value=selected_preset if selected_preset != "Pilih Template Lowongan (Opsional)" else "Data Scientist",
        placeholder="Contoh: Senior Data Scientist"
    )

with col_input_right:
    st.subheader("2. Deskripsi Kebutuhan Pekerjaan (JD)")
    default_jd = PRESET_JOBS.get(selected_preset, "")
    job_desc = st.text_area(
        "Tempelkan kualifikasi atau deskripsi pekerjaan lengkap:",
        value=default_jd,
        height=210,
        placeholder="Tempelkan syarat pekerjaan, tanggung jawab, dan keahlian teknis yang dicari oleh perusahaan di sini..."
    )

st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

# Analyze Action Button
btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
with btn_col2:
    start_analysis = st.button("🚀 Mulai Analisis Kesesuaian Resume", use_container_width=True, type="primary")

if start_analysis:
    if not uploaded_file:
        st.error("⚠️ Silakan unggah berkas resume PDF terlebih dahulu.")
    elif not job_desc.strip():
        st.error("⚠️ Silakan isi deskripsi pekerjaan atau pilih dari contoh template di sidebar.")
    elif not target_pos.strip():
        st.error("⚠️ Mohon cantumkan posisi pekerjaan yang dituju.")
    else:
        with st.spinner("Mengekstrak teks resume dan menjalankan analisis semantik..."):
            try:
                # 1. Extract text from PDF
                pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
                cv_text = ""
                for page in pdf_reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        cv_text += extracted + "\n"
                        
                if not cv_text.strip():
                    st.error("❌ Dokumen PDF tidak memuat teks yang dapat dibaca. Mohon gunakan resume berformat teks digital, bukan hasil foto/scan gambar.")
                else:
                    # 2. Evaluate CV
                    score, matched, missing, breakdown, recommendations = evaluate_cv(
                        cv_text, job_desc, target_pos
                    )
                    
                    st.success("✅ Analisis Resume Berhasil Diselesaikan!")
                    
                    # 3. KPI Metrics Dashboard
                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    
                    with m_col1:
                        color_score = "#10B981" if score >= 75 else ("#F59E0B" if score >= 50 else "#EF4444")
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Tingkat Kecocokan (Overall)</div>
                            <div class="metric-value" style="color: {color_score};">{score}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with m_col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Kecocokan Teknis</div>
                            <div class="metric-value">{breakdown['Technical Skills Match']}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with m_col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Relevansi Konteks</div>
                            <div class="metric-value">{breakdown['Experience & Context Relevance']}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with m_col4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Keterampilan Ditemukan</div>
                            <div class="metric-value">{len(matched)} / {len(matched) + len(missing)}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
                    
                    # 4. Visual Breakdown Chart & Skills Comparison
                    res_col_left, res_col_right = st.columns([1, 1], gap="large")
                    
                    with res_col_left:
                        st.subheader("📊 Analisis Dimensi Kompetensi")
                        
                        # Plotly Horizontal Bar Chart
                        categories = list(breakdown.keys())
                        values = list(breakdown.values())
                        
                        fig = go.Figure(go.Bar(
                            x=values,
                            y=categories,
                            orientation='h',
                            marker=dict(
                                color=['#3B82F6', '#6366F1', '#8B5CF6'],
                                line=dict(color='#1E293B', width=1)
                            ),
                            text=[f"{v}%" for v in values],
                            textposition='auto',
                        ))
                        fig.update_layout(
                            xaxis=dict(range=[0, 100], title="Persentase Pemenuhan (%)"),
                            yaxis=dict(autorange="reversed"),
                            margin=dict(l=20, r=20, t=20, b=20),
                            height=260,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with res_col_right:
                        st.subheader("🔍 Pemetaan Keterampilan")
                        
                        st.markdown("**Keterampilan yang Cocok (Matched):**")
                        if matched:
                            badges_html = "".join([f"<span class='badge-matched'>✓ {s}</span>" for s in matched])
                            st.markdown(badges_html, unsafe_allow_html=True)
                        else:
                            st.info("Belum ada keterampilan spesifik yang secara langsung terdeteksi pada teks resume.")
                            
                        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
                        
                        st.markdown("**Keterampilan yang Belum Muncul (Identified Gaps):**")
                        if missing:
                            missing_html = "".join([f"<span class='badge-missing'>+ {s}</span>" for s in missing])
                            st.markdown(missing_html, unsafe_allow_html=True)
                        else:
                            st.success("Luar biasa! Seluruh kata kunci keterampilan yang dicari telah terpenuhi.")

                    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
                    
                    # 5. Tailored Recommendations
                    st.subheader("💡 Saran Pengembangan & Rekomendasi Resume")
                    for rec in recommendations:
                        st.markdown(f"""
                        <div class="recom-box">
                            {rec}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # 6. CV Text Preview (Collapsible)
                    with st.expander("📄 Tinjau Teks Hasil Ekstraksi Resume"):
                        st.text_area("Isi Teks Resume yang Terdeteksi:", value=cv_text, height=220, disabled=True)
                        
            except Exception as e:
                st.error(f"Terjadi kendala saat memproses berkas resume: {str(e)}")
