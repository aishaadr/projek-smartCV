import os
import resource
import time
from pypdf import PdfReader
from app.model import evaluate_cv, clean_kerning, get_models


# Labeled CV-JD benchmark pairs (expected_match: True for relevant role, False for mismatched role)
BENCHMARK_CASES = [
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
        "cv_path": "data/cv
        _tkj_support.pdf",
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


def run_benchmark(threshold: int = 60) -> None:
    t0 = time.perf_counter()
    models = get_models()
    load_time = time.perf_counter() - t0

    tp = fp = fn = tn = 0
    latencies = []

    print(f"Model load time (CPU): {load_time:.2f}s\n")
    print(f"{'Case':<48} | {'Score':>5} | {'Pred':>5} | {'True':>5} | {'Latency':>7}")
    print("-" * 82)

    for case in BENCHMARK_CASES:
        cv_text = read_pdf_text(case["cv_path"])
        start = time.perf_counter()
        score, matched, missing, _, _, _ = evaluate_cv(
            cv_text, case["jd"], case["target_position"], models=models
        )
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        predicted = score >= threshold
        expected = case["expected_match"]

        if predicted and expected:
            tp += 1
        elif predicted and not expected:
            fp += 1
        elif not predicted and expected:
            fn += 1
        else:
            tn += 1

        print(
            f"{case['name']:<48} | {score:>4}% | {str(predicted):>5} | {str(expected):>5} | {elapsed:>6.3f}s"
        )

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total * 100 if total else 0.0
    precision = tp / (tp + fp) * 100 if (tp + fp) else 0.0
    recall = tp / (tp + fn) * 100 if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    avg_latency = sum(latencies) / len(latencies)
    peak_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

    print("\n=== OFFLINE BENCHMARK SUMMARY ===")
    print(f"Confusion Matrix : TP={tp}, FP={fp}, FN={fn}, TN={tn} (threshold={threshold}%)")
    print(f"Accuracy         : {accuracy:.1f}%")
    print(f"Precision        : {precision:.1f}%")
    print(f"Recall           : {recall:.1f}%")
    print(f"F1-Score         : {f1:.1f}%")
    print(f"Avg CPU Latency  : {avg_latency:.3f}s / CV")
    print(f"Peak RAM Usage   : {peak_ram_mb:.1f} MB")


if __name__ == "__main__":
    run_benchmark()
