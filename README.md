# SmartCV — AI Resume Matcher & Career Analytics Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=flat&logo=streamlit)
![Hugging%20Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?style=flat&logo=huggingface)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688?style=flat&logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**SmartCV** adalah sistem analitik resume cerdas berbasis **Natural Language Processing (NLP)** dan **Semantic Machine Learning**. Aplikasi ini mengevaluasi relevansi resume kandidat terhadap kualifikasi pekerjaan secara kontekstual, melampaui pencarian kata kunci konvensional (*lexical search*). Dilengkapi dengan antarmuka web interaktif berbasis **Streamlit** dan REST API berbasis **FastAPI**.

---

## 🌟 Fitur Utama

- **Contextual Semantic Matching**: Menghitung keselarasan semantik antara profil kandidat dan deskripsi pekerjaan menggunakan model `all-MiniLM-L6-v2` dari **Sentence-Transformers**.
- **Unsupervised Skill Extraction**: Mengekstrak kata kunci kompetensi secara dinamis menggunakan **KeyBERT** tanpa ketergantungan pada kamus statis.
- **Multidimensional Scoring**: Memberikan evaluasi granular dalam 3 dimensi:
  - *Technical Competency Alignment*
  - *Experience & Context Relevance*
  - *Industry Terminology Fit*
- **Actionable Gap Analysis**: Mengidentifikasi keahlian yang belum tercantum (*missing skill sets*) dan memberikan rekomendasi praktis untuk penguatan resume.
- **Modern Interactive Dashboard**: Antarmuka visual yang bersih, intuitif, dan responsif menggunakan **Streamlit** & visualisasi interaktif **Plotly**.
- **FastAPI Microservice Ready**: Endpoint RESTful API berkinerja tinggi untuk integrasi dengan platform web lain.

---

## 🏗️ Arsitektur Proyek

```text
projek-smartCV/
├── streamlit_app.py     # Aplikasi Dashboard Web Interaktif (Streamlit)
├── app/
│   ├── __init__.py
│   ├── main.py          # REST API Server (FastAPI)
│   ├── model.py         # NLP Engine: Sentence-Transformers & KeyBERT
│   └── schemas.py       # Data validation & Pydantic models
├── requirements.txt     # Daftar dependensi Python
├── .gitignore           # File konfigurasi pengecualian Git
└── README.md            # Dokumentasi proyek
```

---

## ⚙️ Persiapan & Instalasi

### 1. Clone Repositori
```bash
git clone https://github.com/aishaadr/projek-smartCV.git
cd projek-smartCV
```

### 2. Buat & Aktifkan Virtual Environment
```bash
# Untuk macOS/Linux:
python3 -m venv venv
source venv/bin/activate

# Untuk Windows:
# python -m venv venv
# venv\Scripts\activate
```

### 3. Instal Dependensi
```bash
pip install -r requirements.txt
```

---

## 🚀 Menjalankan Aplikasi

### Opsi A: Jalankan Dashboard Web Streamlit (Rekomendasi)
Jalankan perintah berikut di terminal:
```bash
streamlit run streamlit_app.py
```
Aplikasi akan otomatis terbuka di peramban pada alamat:  
👉 `http://localhost:8501`

### Opsi B: Jalankan Backend FastAPI Server
Jika ingin menjalankan REST API untuk integrasi sistem:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```
- API Endpoint: `http://localhost:8001`
- Dokumentasi Interaktif Swagger UI: `http://localhost:8001/docs`

---

## 🧠 Metodologi & Alur Kerja NLP

1. **PDF Parsing**: Ekstraksi teks digital dari berkas resume yang diunggah menggunakan `pypdf`.
2. **Text Normalization**: Pembersihan spasi, karakter khusus, dan normalisasi struktur kalimat.
3. **Keyword & Skill Extraction**: Model **KeyBERT** melakukan representasi n-gram teks deskripsi pekerjaan dan mengidentifikasi kata kunci dengan relevansi semantik tertinggi.
4. **Vector Embedding & Cosine Similarity**: Teks resume dan deskripsi lowongan diubah menjadi vektor berdimensi padat (*dense embeddings*) menggunakan encoder `all-MiniLM-L6-v2`. Kemiripan makna dihitung berdasarkan sudut kosinus antar-vektor.
5. **Heuristic Scoring & Synthesis**: Skor dinormalisasi dan disintesis bersama rasio cakupan keahlian teknis untuk menyajikan metrik akhir yang akurat dan berbobot.

---

## 👩‍💻 Pengembang

Dikembangkan secara independen oleh:  
**Aisha Dwi Rahmawati**  
GitHub: [@aishaadr](https://github.com/aishaadr)
