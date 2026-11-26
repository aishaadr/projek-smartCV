import io
import os
import re

# Prevent tokenizer multithreading deadlocks on macOS/Linux
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import plotly.graph_objects as go
from pypdf import PdfReader

from app.model import evaluate_cv, clean_kerning, get_models

# ==========================================
# PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="SmartCV Analytics — AI Resume & Job Matcher",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Cache model loading across Streamlit reruns
@st.cache_resource(show_spinner="Memuat model NLP (all-MiniLM-L6-v2 & KeyBERT)...")
def load_cached_models():
    return get_models()

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
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #94A3B8;
        margin-top: 0.2rem;
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
    .badge-surplus {
        display: inline-block;
        background-color: #EFF6FF;
        color: #1E40AF;
        border: 1px solid #BFDBFE;
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
# PRESET JOB SAMPLES (SMA/SMK & ENTRY-LEVEL)
# ==========================================
PRESET_JOBS = {
    "Pilih Template Lowongan (Opsional)": "",
    "Junior Web Developer (RPL)": """Kami mencari Junior Web Developer / lulusan SMK RPL yang siap membangun dan memelihara website modern.
Tanggung Jawab Utama:
- Mengembangkan tampilan antarmuka web responsif menggunakan HTML, CSS, JavaScript, dan Tailwind CSS atau Bootstrap.
- Membangun fitur backend sederhana menggunakan PHP (Laravel / CodeIgniter) atau Node.js dan database MySQL.
- Berkolaborasi menggunakan Git dan GitHub serta mengintegrasikan REST API.
Kualifikasi:
- Lulusan SMK RPL atau mahasiswa/fresh graduate bidang informatika.
- Menguasai HTML, CSS, JavaScript, PHP, MySQL, dan Git.
- Memiliki kemampuan problem solving, teamwork, dan komunikasi yang baik.""",

    "IT Support / Teknisi Jaringan (TKJ)": """Dibutuhkan IT Support / Network Technician (lulusan SMK TKJ dipersilakan melamar) untuk mendukung operasional infrastruktur IT.
Tanggung Jawab Utama:
- Melakukan instalasi jaringan komputer LAN/WAN, crimping kabel, serta konfigurasi router MikroTik dan Cisco.
- Menangani hardware troubleshooting, software troubleshooting, instalasi OS Windows Server maupun Linux.
- Memberikan layanan helpdesk melalui ticketing system dan menjaga keamanan jaringan (Firewall).
Kualifikasi:
- Lulusan SMK TKJ atau sederajat.
- Memahami dasar Networking, TCP/IP, MikroTik, Hardware Troubleshooting, dan Linux.
- Memiliki komunikasi yang baik, problem solving, dan manajemen waktu.""",

    "UI/UX & Desain Grafis (DKV)": """Kami membutuhkan Junior UI/UX Designer & Graphic Designer (lulusan SMK DKV / Multimedia) untuk merancang tampilan aplikasi dan materi visual.
Tanggung Jawab Utama:
- Membuat wireframing, prototyping, user flow, dan UI Design interaktif menggunakan Figma.
- Mendesain aset visual, poster, dan konten digital menggunakan Adobe Photoshop, Adobe Illustrator, Canva, atau CorelDraw.
- Melakukan UX Research sederhana dan usability testing bersama tim produk.
Kualifikasi:
- Menguasai Figma, Adobe Photoshop, Adobe Illustrator, dan prinsip Typography.
- Memahami Design System, Wireframing, dan Prototyping.
- Kreatif, komunikatif, dan mampu bekerja sama dalam tim.""",

    "Admin Data & Operasional": """Dicari Staff Admin Data untuk mengelola input, pembersihan, dan pelaporan data operasional harian.
Tanggung Jawab Utama:
- Melakukan data entry dan data cleaning secara akurat menggunakan Microsoft Excel, Google Sheets, dan Microsoft Word.
- Membuat laporan mingguan menggunakan Pivot Table, VLOOKUP, serta dashboard sederhana di Google Data Studio.
- Mengelola basis data dasar menggunakan SQL.
Kualifikasi:
- Lulusan SMA/SMK sederajat dengan ketelitian tinggi.
- Mahir menggunakan Microsoft Excel (Pivot Table, VLOOKUP), Microsoft Word, dan Google Sheets.
- Memiliki kemampuan critical thinking, time management, dan reporting yang rapi.""",

    "Junior Data Scientist": """Kami mencari Junior Data Scientist yang memiliki pemahaman kuat dalam pengembangan solusi Machine Learning dan Natural Language Processing (NLP) secara end-to-end.
Tanggung Jawab Utama:
- Mengembangkan pipeline Machine Learning dan Deep Learning untuk klasifikasi, regresi, clustering, dan sentiment analysis.
- Melakukan Exploratory Data Analysis (EDA), feature engineering, menangani class imbalance, serta stratified cross-validation untuk mencegah data leakage.
- Mengemas model prediktif ke dalam aplikasi interaktif berbasis Streamlit.
Kualifikasi:
- Menguasai Python, TensorFlow, PyTorch, Keras, SQL, dan Microsoft Excel.
- Memahami Supervised Learning, Unsupervised Learning, Time Series, dan Business Analysis.
- Memiliki kemampuan critical thinking, problem solving, communication skills, dan leadership.""",
}


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1586281380349-632531db7ed4?w=400&auto=format&fit=crop&q=80", use_container_width=True)
    st.markdown("### ⚙️ Konfigurasi Lowongan")
    
    selected_preset = st.selectbox(
        "Gunakan Contoh Posisi Pekerjaan:",
        options=list(PRESET_JOBS.keys()),
    )
    
    st.markdown("---")
    st.markdown("### 🧠 Arsitektur Model")
    st.markdown("""
    - **Embedder**: `all-MiniLM-L6-v2`  
      *(Hugging Face Sentence-Transformers)*
    - **Skill Extractor**: Hybrid (Industry Taxonomy + Bilingual Synonyms + KeyBERT)
    - **Language Support**: Dwibahasa (Bahasa Indonesia & English)
    - **Execution Device**: CPU (Stabil & Responsif)
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
    <p class="hero-subtitle">Platform analitik pencocokan resume terhadap kualifikasi pekerjaan berbasis Semantic NLP & Machine Learning.</p>
</div>
""", unsafe_allow_html=True)

col_input_left, col_input_right = st.columns([1, 1], gap="large")

with col_input_left:
    st.subheader("1. Unggah Berkas Resume (CV)")
    uploaded_file = st.file_uploader(
        "Pilih file resume dalam format PDF:",
        type=["pdf"],
        help="Pastikan PDF memuat teks digital yang dapat dibaca.",
    )
    
    initial_pos = selected_preset if selected_preset != "Pilih Template Lowongan (Opsional)" else "Junior Web Developer"
    target_pos = st.text_input(
        "Posisi Pekerjaan yang Dituju:",
        value=initial_pos,
        placeholder="Contoh: Junior Web Developer",
    )

with col_input_right:
    st.subheader("2. Deskripsi Kebutuhan Pekerjaan (JD)")
    default_jd = PRESET_JOBS.get(selected_preset, "")
    job_desc = st.text_area(
        "Tempelkan kualifikasi atau deskripsi pekerjaan lengkap:",
        value=default_jd,
        height=210,
        placeholder="Tempelkan syarat pekerjaan, tanggung jawab, dan keahlian teknis yang dicari di sini...",
    )

st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

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
        models = load_cached_models()
        with st.spinner("Mengekstrak teks resume dan menjalankan analisis semantik..."):
            try:
                pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
                cv_text = ""
                for page in pdf_reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        cv_text += extracted + "\n"
                        
                cv_text = clean_kerning(cv_text)
                
                if not cv_text.strip():
                    st.error("❌ Dokumen PDF tidak memuat teks yang dapat dibaca. Mohon gunakan resume berformat teks digital, bukan hasil foto/scan gambar.")
                else:
                    score, matched, missing, breakdown, recommendations, metrics = evaluate_cv(
                        cv_text, job_desc, target_pos, models=models
                    )
                    
                    st.success("✅ Analisis Resume Berhasil Diselesaikan!")
                    
                    # Honest recruitment metrics in the 4 KPI cards
                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    
                    with m_col1:
                        color_score = "#10B981" if score >= 75 else ("#F59E0B" if score >= 50 else "#EF4444")
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Skor Kecocokan</div>
                            <div class="metric-value" style="color: {color_score};">{score}%</div>
                            <div class="metric-sub">Keselarasan Keseluruhan</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with m_col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Skill Wajib Terpenuhi</div>
                            <div class="metric-value" style="color: #10B981;">{metrics['matched_count']} / {metrics['total_required']}</div>
                            <div class="metric-sub">Kualifikasi Inti Lowongan</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with m_col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Kemiripan Konteks</div>
                            <div class="metric-value" style="color: #3B82F6;">{metrics['cosine_similarity']}%</div>
                            <div class="metric-sub">Semantic Cosine Similarity</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with m_col4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Skill Nilai Plus</div>
                            <div class="metric-value" style="color: #8B5CF6;">{metrics['surplus_count']}</div>
                            <div class="metric-sub">Keahlian Ekstra Kandidat</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
                    
                    # Visual Breakdown Chart & Skills Comparison
                    res_col_left, res_col_right = st.columns([1, 1], gap="large")
                    
                    with res_col_left:
                        st.subheader("📊 Analisis Dimensi Kompetensi")
                        
                        categories = [
                            "Kecocokan Keterampilan Teknis",
                            "Relevansi Konteks Pengalaman",
                            "Keselarasan Terminologi Industri",
                        ]
                        values = [
                            breakdown["technical"],
                            breakdown["experience"],
                            breakdown["industry"],
                        ]
                        
                        fig = go.Figure(go.Bar(
                            x=values,
                            y=categories,
                            orientation="h",
                            marker=dict(
                                color=["#10B981", "#3B82F6", "#8B5CF6"],
                                line=dict(color="#1E293B", width=1),
                            ),
                            text=[f"{v}%" for v in values],
                            textposition="auto",
                        ))
                        fig.update_layout(
                            xaxis=dict(range=[0, 100], title="Persentase Pemenuhan (%)"),
                            yaxis=dict(autorange="reversed"),
                            margin=dict(l=20, r=20, t=20, b=20),
                            height=240,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with res_col_right:
                        st.subheader("🔍 Pemetaan Keterampilan")
                        
                        st.markdown(f"**Keterampilan Wajib Terpenuhi ({len(matched)} Terdeteksi):**")
                        if matched:
                            badges_html = "".join([f"<span class='badge-matched'>✓ {s}</span>" for s in matched])
                            st.markdown(badges_html, unsafe_allow_html=True)
                        else:
                            st.info("Belum ada keterampilan spesifik yang secara langsung terdeteksi pada teks resume.")
                            
                        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                        
                        st.markdown(f"**Skill Nilai Plus / Tambahan ({len(metrics['surplus_skills'])} Teridentifikasi):**")
                        if metrics["surplus_skills"]:
                            surplus_html = "".join([f"<span class='badge-surplus'>★ {s}</span>" for s in metrics["surplus_skills"][:8]])
                            st.markdown(surplus_html, unsafe_allow_html=True)
                            
                        if missing:
                            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                            st.markdown(f"**Keterampilan Gap / Belum Muncul ({len(missing)} Teridentifikasi):**")
                            missing_html = "".join([f"<span class='badge-missing'>+ {s}</span>" for s in missing])
                            st.markdown(missing_html, unsafe_allow_html=True)

                    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
                    
                    # Tailored Recommendations (convert **bold** markdown to HTML <strong>)
                    st.subheader("💡 Saran Pengembangan & Rekomendasi Resume")
                    for rec in recommendations:
                        rec_html = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", rec)
                        st.markdown(f"""
                        <div class="recom-box">
                            {rec_html}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with st.expander("📄 Tinjau Teks Hasil Ekstraksi Resume"):
                        st.text_area("Isi Teks Resume yang Terdeteksi:", value=cv_text, height=220, disabled=True)
                        
            except Exception as e:
                st.error(f"Terjadi kendala saat memproses berkas resume: {str(e)}")
