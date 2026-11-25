# SmartCV - AI & Data Science Microservice

Microservice pemrosesan dan analisis kecocokan resume (CV) terhadap deskripsi lowongan pekerjaan (Job Description) menggunakan **Hugging Face Sentence-Transformers** (`all-MiniLM-L6-v2`) dan **KeyBERT**.

## Fitur Utama
- **Semantic Cosine Similarity**: Mengukur kemiripan makna antara CV dan kualifikasi lowongan.
- **Dynamic Skill Extraction**: Ekstraksi keterampilan menggunakan KeyBERT tanpa ketergantungan kamus manual statis.
- **Smart Feedback & Recommendations**: Menghasilkan rekomendasi personal berdasarkan keterampilan yang kurang (*missing skills*).
- **FastAPI Endpoint**: API berkecepatan tinggi yang siap diintegrasikan dengan aplikasi web (Laravel/Web-Bengkod).

## Cara Menjalankan

1. Buat & aktifkan virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Instal dependensi:
```bash
pip install -r requirements.txt
```

3. Jalankan server FastAPI:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

Server akan aktif di `http://localhost:8001` (Dokumentasi Swagger UI di `http://localhost:8001/docs`).
