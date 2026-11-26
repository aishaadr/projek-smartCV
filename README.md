# SmartCV — AI Resume Matcher & Career Analytics Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=flat&logo=streamlit)
![Hugging%20Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?style=flat&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**SmartCV** adalah sistem analitik resume berbasis **Natural Language Processing (NLP)** dan **Semantic Machine Learning**. Aplikasi ini mengevaluasi kesesuaian resume kandidat terhadap deskripsi pekerjaan secara kontekstual (dwibahasa Indonesia & Inggris), dilengkapi antarmuka web interaktif berbasis **Streamlit** dan skrip **benchmark offline** untuk mengukur akurasi serta efisiensi CPU/RAM.

---

## 🌟 Fitur Utama

- **Contextual Semantic Matching**: Menghitung kemiripan makna antara resume dan deskripsi lowongan menggunakan model multilingual `paraphrase-multilingual-MiniLM-L12-v2` dari **Sentence-Transformers** di CPU secara lokal.
- **Hybrid Skill Extraction**: Menggabungkan taksonomi kompetensi industri (AI/Data, Web/RPL, IT Support/TKJ, UI/UX-DKV, Admin Data), pemetaan sinonim dwibahasa (ID/EN), serta **KeyBERT**.
- **Metrik Rekrutmen Objektif**: Menampilkan Skor Kecocokan, Skill Wajib Terpenuhi (`X/Y`), Kemiripan Konteks (*Cosine Similarity*), dan Skill Nilai Plus.
- **Actionable Gap Analysis**: Mengidentifikasi keahlian yang belum tercantum (*missing skills*) beserta saran pengembangan resume yang spesifik.
- **Offline Benchmark Suite**: Tersedia `benchmark.py` dan sampel CV anonim di folder `data/` untuk menguji akurasi, F1-Score, latensi CPU, dan konsumsi RAM secara terpisah.

---

## 🏗️ Arsitektur Proyek

```text
projek-smartCV/
├── streamlit_app.py     # Aplikasi Dashboard Web Interaktif (Streamlit)
├── benchmark.py         # Skrip Benchmark Offline (Accuracy, F1, CPU & RAM)
├── app/
│   ├── __init__.py
│   └── model.py         # NLP Engine: Sentence-Transformers & KeyBERT
├── data/                # Sampel CV PDF anonim untuk pengujian & benchmark
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
python3 -m venv venv
source venv/bin/activate
```

### 3. Instal Dependensi
```bash
pip install -r requirements.txt
```

---

## 🚀 Menjalankan Aplikasi & Benchmark

### Jalankan Dashboard Web Streamlit
```bash
streamlit run streamlit_app.py
```
Aplikasi akan terbuka di browser pada alamat `http://localhost:8501`.

### Jalankan Benchmark Offline
```bash
python3 benchmark.py
```

---

## 👩‍💻 Pengembang

Dikembangkan oleh:  
**Aisha Dwi Rahmawati**  
GitHub: [@aishaadr](https://github.com/aishaadr)
