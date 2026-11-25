import io
import os
import re

# Cegah konflik tokenizers multithreading pada macOS/Linux
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import plotly.graph_objects as go
from pypdf import PdfReader

# Import model & engine dari app.model
from app.model import evaluate_cv, clean_kerning

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
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem 0.3rem;
    }
    .badge-missing {
        display: inline-block;
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem 0.3rem;
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
# PRESET JOB SAMPLES FOR EASY TESTING
# ==========================================
PRESET_JOBS = {
    "Pilih Template Lowongan (Opsional)": "",
    "Junior Data Scientist / Data Science Intern": """Tentang Posisi Ini (Deskripsi Panjang)
Kami mencari Junior Data Scientist / Data Science Intern yang memiliki pemahaman teknis kuat dalam membangun solusi Machine Learning dan Natural Language Processing (NLP) secara end-to-end. Dalam peran ini, kamu akan bertugas mengolah dataset dunia nyata—mulai dari data tabular keuangan, klinis, hingga teks media sosial—menjadi model prediktif yang siap produksi (production-ready).
Kamu akan terlibat penuh dalam seluruh siklus hidup data: menangani masalah class imbalance, mencegah data leakage, melakukan feature engineering yang berdampak pada bisnis, hingga melakukan deployment model menjadi aplikasi interaktif.
Tanggung Jawab Utama
• Pengembangan Pipeline ML & Deep Learning: Merancang, melatih, dan mengevaluasi model klasifikasi, regresi, clustering, dan NLP untuk berbagai kasus penggunaan (seperti segmentasi pelanggan, risk scoring, dan analisis sentimen).
• Rekayasa Fitur & Validasi Ketat: Melakukan feature engineering tingkat lanjut, menangani dataset dengan ketimpangan kelas (severe class imbalance), serta menerapkan validasi silang (stratified cross-validation) untuk memastikan model bebas dari overfitting dan data leakage.
• Pengolahan Teks & NLP: Membangun sistem pemrosesan bahasa alami untuk data teks Bahasa Indonesia, termasuk klasifikasi sentimen dan pemeriksaan tata bahasa/ejaan kontekstual.
• Deployment Aplikasi Data: Mengemas model Machine Learning ke dalam aplikasi web interaktif (seperti menggunakan Streamlit) agar dapat diakses dan digunakan secara real-time oleh pengguna maupun pemangku kepentingan.
• Penerjemahan Wawasan Bisnis: Mengubah hasil klasterisasi dan probabilitas prediksi menjadi kerangka kerja segmentasi risiko serta strategi bisnis yang dapat ditindaklanjuti (actionable recommendations).
Keahlian & Kualifikasi yang Dibutuhkan
• Latar Belakang Akademik: Mahasiswa aktif atau lulusan S1 Teknik Informatika, Ilmu Komputer, atau bidang kuantitatif terkait dengan rekam jejak akademik yang kuat (IPK > 3.50 menjadi nilai tambah) dan pengalaman proyek data nyata.
• Bahasa Pemrograman & Framework: Menguasai Python serta ekosistem Machine Learning dan Deep Learning seperti TensorFlow, PyTorch, dan Keras, serta mahir menggunakan SQL dan Microsoft Excel untuk manipulasi serta analisis data.
• Keahlian Teknis Data Science: Pemahaman mendalam mengenai Supervised & Unsupervised Learning, Time Series, Exploratory Data Analysis (EDA), evaluasi metrik model (Accuracy, Recall, F1-Score, pengurangan error terhadap baseline), serta pemecahan masalah imbalanced dataset.
• Pengalaman Deployment & Analisis Bisnis: Terbukti pernah membangun dan men-deploy aplikasi berbasis ML (misalnya dengan Streamlit) serta memiliki kemampuan Business Analysis untuk menyelaraskan solusi teknis dengan kebutuhan pengguna.
• Soft Skills: Memiliki kemampuan berpikir kritis (critical thinking), pemecahan masalah (problem solving), kepemimpinan, kerja sama tim, dan komunikasi yang baik untuk mempresentasikan temuan data.""",

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
    - **Skill Extractor**: Hybrid (Industry Taxonomy + Bilingual Synonyms + KeyBERT)
    - **Language Support**: Dwibahasa (Bahasa Indonesia & English)
    - **Execution Device**: CPU (Stabil & Cepat)
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
        help="Pastikan PDF memuat teks digital yang dapat dibaca."
    )
    
    # Auto-fill target position if preset is selected
    initial_pos = selected_preset if selected_preset != "Pilih Template Lowongan (Opsional)" else "Junior Data Scientist"
    target_pos = st.text_input(
        "Posisi Pekerjaan yang Dituju:",
        value=initial_pos,
        placeholder="Contoh: Junior Data Scientist"
    )

with col_input_right:
    st.subheader("2. Deskripsi Kebutuhan Pekerjaan (JD)")
    default_jd = PRESET_JOBS.get(selected_preset, "")
    job_desc = st.text_area(
        "Tempelkan kualifikasi atau deskripsi pekerjaan lengkap:",
        value=default_jd,
        height=210,
        placeholder="Tempelkan syarat pekerjaan, tanggung jawab, dan keahlian teknis yang dicari di sini..."
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
                        
                # Normalisasi kerning text PDF
                cv_text = clean_kerning(cv_text)
                
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
                            <div class="metric-value">{breakdown['technical']}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with m_col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Relevansi Pengalaman</div>
                            <div class="metric-value">{breakdown['experience']}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with m_col4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Keterampilan Terpenuhi</div>
                            <div class="metric-value">{len(matched)} / {len(matched) + len(missing)}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
                    
                    # 4. Visual Breakdown Chart & Skills Comparison
                    res_col_left, res_col_right = st.columns([1, 1], gap="large")
                    
                    with res_col_left:
                        st.subheader("📊 Analisis Dimensi Kompetensi")
                        
                        categories = [
                            "Kecocokan Keterampilan Teknis",
                            "Relevansi Konteks Pengalaman",
                            "Keselarasan Terminologi Industri"
                        ]
                        values = [
                            breakdown['technical'],
                            breakdown['experience'],
                            breakdown['industry']
                        ]
                        
                        fig = go.Figure(go.Bar(
                            x=values,
                            y=categories,
                            orientation='h',
                            marker=dict(
                                color=['#10B981', '#3B82F6', '#8B5CF6'],
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
                        
                        st.markdown(f"**Keterampilan yang Cocok ({len(matched)} Terdeteksi):**")
                        if matched:
                            badges_html = "".join([f"<span class='badge-matched'>✓ {s}</span>" for s in matched])
                            st.markdown(badges_html, unsafe_allow_html=True)
                        else:
                            st.info("Belum ada keterampilan spesifik yang secara langsung terdeteksi pada teks resume.")
                            
                        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
                        
                        st.markdown(f"**Keterampilan Tambahan / Gap ({len(missing)} Teridentifikasi):**")
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
