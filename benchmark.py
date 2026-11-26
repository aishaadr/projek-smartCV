import resource
import time
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from app.model import evaluate_cv, clean_kerning, get_models

# Real CV text provided by the user (Education, Skills, Projects, Achievement, Certifications)
AISHA_REAL_CV = """
Informatics Engineering student (GPA 3.86) at Universitas Dian Nuswantoro, selected as Unggulan Class
Batch 2024 (top 40 students), specializing in Data Science with hands-on experience building end-to-end
ML and NLP pipelines using Python, TensorFlow, PyTorch, and Keras. Built and deployed models across
healthcare and finance domains, including a child stunting classifier (99.81% accuracy, 99% recall across
5,236 records) and a credit card segmentation system (97.77% accuracy across 8,950 records).
Experienced in handling real-world data challenges including severe class imbalance, leakage prevention,
feature engineering, and model deployment.
EDUCATION
UNIVERSITAS DIAN NUSWANTORO Informatics Engineering (IPK: 3.86) Agustus 2024 - Present
Relevant Coursework: Data Mining, Database Systems, Data Structures & Algorithms, Human-Computer Interaction
SMAN 2 KUDUS Head of Chemistry Club July 2021 - May 2024
SOFT SKILLS: Critical Thinking, Problem Solving, Communication Skills, Collaboration, Curiosity, Time Management, Adaptability, Leadership
TECHNICAL SKILLS: Machine Learning, TensorFlow, Keras, PyTorch, SQL, Microsoft Excel, Deep Learning, Python
Projects:
Customer Segmentation for Data-Driven Marketing: Built a two-stage machine learning system identifying four distinct behavioral customer profiles from 8,950 credit card records with 97.77% accuracy.
Child Stunting Prediction: Built a pediatric screening pipeline across 5,236 clinical records achieving 99.81% accuracy and 99% recall, validated through 5-fold stratified cross-validation.
Automatic Proofreading System for Indonesian Academic Documents: Built an intelligent proofreading system enforcing EYD V language standardization, deployed as a Streamlit web application.
Sentiment Analysis of Public Perception on the MBG Program: Built a three-class sentiment classifier on 3,263 YouTube comments achieving 87% accuracy, deployed as a real-time Streamlit web application.
UMKM Success Probability Prediction: Predicted MSME business success across 250 imbalanced records, engineering 8 interaction features, 49.6% error reduction over baseline, 4-tier risk segmentation framework.
Achievement: Data Science Competition, Informatics Festival 2026 | Top 5 of 52 teams nationally.
Certifications: CSS Website Design, HTML 5 Basic, Supervised Learning Regression (IBM), Time Series and Survival Analysis (IBM), Machine Learning Specialization (IBM), Exploratory Data Analysis (IBM), Deep Learning and Reinforcement Learning (IBM), Unsupervised Machine Learning (IBM), Supervised Machine Learning Classification (IBM), Requirements Gathering in Business Analysis (Microsoft), Power Platform in Business Analysis (Microsoft), Microsoft Business Analyst (Microsoft), Data for Business Analysts Using Microsoft Excel (Microsoft), Microsoft Visio (Microsoft), Deep Learning with Keras and TensorFlow (IBM), Deep Learning with PyTorch (IBM).
"""

BENCHMARK_CASES = [
    {
        "name": "CV Asli Aisha -> Junior Data Scientist",
        "cv_text": AISHA_REAL_CV,
        "target_position": "Junior Data Scientist",
        "jd": (
            "We are seeking a Junior Data Scientist / Data Science Intern to build end-to-end "
            "Machine Learning, Deep Learning, and Natural Language Processing (NLP) pipelines. "
            "Responsibilities: classification, regression, clustering, customer segmentation, "
            "Exploratory Data Analysis (EDA), feature engineering, class imbalance, data leakage, "
            "stratified cross-validation, sentiment analysis, and Streamlit deployment. "
            "Required: Python, SQL, Microsoft Excel, TensorFlow, PyTorch, Keras, Supervised Learning, "
            "Unsupervised Learning, Business Analysis, critical thinking, problem solving, leadership."
        ),
        "expected_match": True,
    },
    {
        "name": "CV Asli Aisha -> ML & NLP Engineer (ID)",
        "cv_text": AISHA_REAL_CV,
        "target_position": "ML & NLP Engineer",
        "jd": (
            "Dicari Machine Learning & NLP Engineer untuk membangun model prediktif dan pengolahan teks "
            "(Natural Language Processing). Menguasai Python, TensorFlow, PyTorch, Keras, Deep Learning, "
            "Machine Learning, feature engineering, class imbalance, serta deployment Streamlit."
        ),
        "expected_match": True,
    },
    {
        "name": "CV Asli Aisha -> IT Support / TKJ (Mismatch)",
        "cv_text": AISHA_REAL_CV,
        "target_position": "IT Support",
        "jd": (
            "Dibutuhkan Teknisi Jaringan TKJ. Menguasai MikroTik, Cisco, Crimping, LAN/WAN, "
            "Hardware Troubleshooting, Software Troubleshooting, Windows Server, Active Directory, "
            "Ticketing System, Helpdesk, dan Firewall."
        ),
        "expected_match": False,
    },
    {
        "name": "CV Asli Aisha -> UI/UX & DKV (Mismatch)",
        "cv_text": AISHA_REAL_CV,
        "target_position": "UI/UX Designer",
        "jd": (
            "Dicari UI/UX & Graphic Designer DKV. Menguasai Figma, Adobe Photoshop, Adobe Illustrator, "
            "CorelDraw, Canva, Wireframing, Prototyping, Design System, User Flow, Usability Testing, "
            "dan Typography."
        ),
        "expected_match": False,
    },
    {
        "name": "RPL CV -> Junior Web Developer JD",
        "cv_path": "data/cv_rpl_web.pdf",
        "target_position": "Junior Web Developer",
        "jd": (
            "Kami mencari Junior Web Developer / lulusan SMK RPL. Menguasai HTML, CSS, JavaScript, "
            "PHP, Laravel, Tailwind CSS, Bootstrap, MySQL, REST API, Git, dan GitHub. "
            "Memiliki kemampuan problem solving dan teamwork."
        ),
        "expected_match": True,
    },
    {
        "name": "TKJ CV -> IT Support / Network Technician JD",
        "cv_path": "data/cv_tkj_support.pdf",
        "target_position": "IT Support",
        "jd": (
            "Dibutuhkan IT Support / Teknisi Jaringan lulusan SMK TKJ. Menguasai Networking, TCP/IP, "
            "MikroTik, Cisco, LAN/WAN, Hardware Troubleshooting, Software Troubleshooting, Linux, "
            "Windows Server, Helpdesk, dan Firewall."
        ),
        "expected_match": True,
    },
    {
        "name": "DKV CV -> UI/UX & Graphic Designer JD",
        "cv_path": "data/cv_dkv_uiux.pdf",
        "target_position": "UI/UX Designer",
        "jd": (
            "Dicari Junior UI/UX & Graphic Designer lulusan SMK DKV. Mahir menggunakan Figma, "
            "Adobe Photoshop, Adobe Illustrator, Canva, Wireframing, Prototyping, UI Design, "
            "UX Research, Design System, dan Typography."
        ),
        "expected_match": True,
    },
    {
        "name": "Admin Data CV -> Staff Admin Data JD",
        "cv_path": "data/cv_admin_data.pdf",
        "target_position": "Admin Data",
        "jd": (
            "Dicari Staff Admin Data SMA/SMK. Menguasai Microsoft Excel, Pivot Table, VLOOKUP, "
            "Google Sheets, Microsoft Word, Data Entry, Data Cleaning, SQL, dan Reporting."
        ),
        "expected_match": True,
    },
    {
        "name": "RPL CV -> IT Support JD (Mismatch)",
        "cv_path": "data/cv_rpl_web.pdf",
        "target_position": "IT Support",
        "jd": (
            "Dibutuhkan Teknisi Jaringan TKJ. Menguasai MikroTik, Cisco, Crimping, LAN/WAN, "
            "Hardware Troubleshooting, Windows Server, Active Directory, dan Firewall."
        ),
        "expected_match": False,
    },
    {
        "name": "TKJ CV -> UI/UX Designer JD (Mismatch)",
        "cv_path": "data/cv_tkj_support.pdf",
        "target_position": "UI/UX Designer",
        "jd": (
            "Dicari UI/UX Designer. Menguasai Figma, Adobe Photoshop, Adobe Illustrator, "
            "Wireframing, Prototyping, Design System, User Flow, dan Typography."
        ),
        "expected_match": False,
    },
    {
        "name": "DKV CV -> Admin Data JD (Mismatch)",
        "cv_path": "data/cv_dkv_uiux.pdf",
        "target_position": "Admin Data",
        "jd": (
            "Dicari Admin Data. Mahir Microsoft Excel, Pivot Table, VLOOKUP, SQL, "
            "Data Entry, Data Cleaning, dan Google Sheets."
        ),
        "expected_match": False,
    },
    {
        "name": "Admin Data CV -> Junior Web Developer JD (Mismatch)",
        "cv_path": "data/cv_admin_data.pdf",
        "target_position": "Junior Web Developer",
        "jd": (
            "Dicari Junior Web Developer. Menguasai HTML, CSS, JavaScript, PHP, Laravel, "
            "React, Docker, Git, dan REST API."
        ),
        "expected_match": False,
    },
]


def read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return clean_kerning(text)


def baseline_tfidf_score(cv_text: str, jd_text: str) -> float:
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([clean_kerning(cv_text), clean_kerning(jd_text)])
    sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    return round(sim * 100, 1)


def summarize_pure_metrics(scores: list, labels: list) -> dict:
    pos_scores = [s for s, y in zip(scores, labels) if y]
    neg_scores = [s for s, y in zip(scores, labels) if not y]
    avg_pos = sum(pos_scores) / len(pos_scores)
    avg_neg = sum(neg_scores) / len(neg_scores)
    sep_gap = avg_pos - avg_neg
    # Pure continuous accuracy: 100 - Mean Absolute Error against ideal (100% for match, 0% for mismatch)
    mae = sum(abs((100.0 if y else 0.0) - s) for s, y in zip(scores, labels)) / len(scores)
    pure_accuracy = 100.0 - mae
    roc_auc = roc_auc_score([1 if y else 0 for y in labels], scores) * 100
    return {
        "avg_pos": avg_pos,
        "avg_neg": avg_neg,
        "sep_gap": sep_gap,
        "mae": mae,
        "pure_accuracy": pure_accuracy,
        "roc_auc": roc_auc,
    }


def run_benchmark() -> None:
    t0 = time.perf_counter()
    models = get_models()
    load_time = time.perf_counter() - t0

    scv_scores = []
    tf_scores = []
    labels = []
    scv_latencies = []
    tf_latencies = []

    print(f"SmartCV model load time (CPU): {load_time:.2f}s\n")
    print(f"{'Case':<48} | {'SmartCV':>7} | {'TF-IDF':>6} | {'Target':>7}")
    print("-" * 77)

    for case in BENCHMARK_CASES:
        cv_text = case["cv_text"] if "cv_text" in case else read_pdf_text(case["cv_path"])
        expected = case["expected_match"]
        labels.append(expected)

        s1 = time.perf_counter()
        scv_score, _, _, _, _, _ = evaluate_cv(
            cv_text, case["jd"], case["target_position"], models=models
        )
        scv_latencies.append(time.perf_counter() - s1)
        scv_scores.append(float(scv_score))

        s2 = time.perf_counter()
        tf_score = baseline_tfidf_score(cv_text, case["jd"])
        tf_latencies.append(time.perf_counter() - s2)
        tf_scores.append(tf_score)

        target_label = "Cocok" if expected else "Beda"
        print(
            f"{case['name']:<48} | {scv_score:>6}% | {tf_score:>5.1f}% | {target_label:>7}"
        )

    scv_m = summarize_pure_metrics(scv_scores, labels)
    tf_m = summarize_pure_metrics(tf_scores, labels)
    peak_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

    print("\n=== PURE BENCHMARK COMPARISON (NO THRESHOLD, 12 CASES) ===")
    print(f"{'Metric':<28} | {'Baseline (TF-IDF)':>17} | {'SmartCV (HF + Hybrid)':>21}")
    print("-" * 72)
    print(f"{'Avg Score (CV Cocok)':<28} | {tf_m['avg_pos']:>16.1f}% | {scv_m['avg_pos']:>20.1f}%")
    print(f"{'Avg Score (CV Beda Bidang)':<28} | {tf_m['avg_neg']:>16.1f}% | {scv_m['avg_neg']:>20.1f}%")
    print(f"{'Separation Gap (Selisih)':<28} | {tf_m['sep_gap']:>16.1f}% | {scv_m['sep_gap']:>20.1f}%")
    print(f"{'Pure Accuracy (100 - MAE)':<28} | {tf_m['pure_accuracy']:>16.1f}% | {scv_m['pure_accuracy']:>20.1f}%")
    print(f"{'ROC-AUC (Ranking Pure)':<28} | {tf_m['roc_auc']:>16.1f}% | {scv_m['roc_auc']:>20.1f}%")
    print(f"{'Avg CPU Latency':<28} | {sum(tf_latencies)/len(tf_latencies):>15.3f}s | {sum(scv_latencies)/len(scv_latencies):>19.3f}s")
    print(f"{'Peak RAM (Process)':<28} | {'-':>17} | {peak_ram_mb:>18.1f} MB")


if __name__ == "__main__":
    run_benchmark()
