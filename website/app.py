"""
app.py - Website prediksi kanker payudara (Flask).

Pipeline: 30 fitur --[Grey Wolf Optimizer]--> 8 fitur --[Random Forest]--> diagnosis

Menjalankan:
    pip install -r requirements.txt
    python train_model.py        (sekali saja, menghasilkan model/*.pkl)
    python app.py
    buka http://127.0.0.1:5000

Route:
    GET  /                 landing page
    GET  /diagnostics      halaman analisis (form 8 fitur)
    POST /predict          JSON  {fitur: nilai, ...}  ->  hasil prediksi
    GET  /api/sample       ambil satu contoh kasus dari dataset (?type=malignant|benign)
    GET  /api/metrics      metrik model
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

try:                                        # scikit-learn >= 1.3
    from sklearn.exceptions import InconsistentVersionWarning
except ImportError:                         # versi lama
    InconsistentVersionWarning = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "dataset_bersih.csv")

# Template memakai folder "template" (bukan "templates" default Flask),
# supaya struktur folder project tidak perlu diubah.
app = Flask(__name__, template_folder="template", static_folder="static")

LABEL = {0: "Benign", 1: "Malignant"}
LABEL_ID = {0: "Jinak", 1: "Ganas"}


# --------------------------------------------------------------------------
# Pemuatan model
# --------------------------------------------------------------------------
def _muat_pickle(path, ketat=True):
    """Muat pickle. Kalau ketat=True, beda versi scikit-learn dianggap ERROR.

    Kalau model dibuat oleh versi scikit-learn lain, sklearn hanya memberi
    peringatan lalu tetap memuatnya - hasilnya bisa diam-diam salah. Karena itu
    peringatan tersebut dinaikkan jadi error, supaya bisa ditangani.
    """
    with warnings.catch_warnings():
        if ketat and InconsistentVersionWarning is not None:
            warnings.simplefilter("error", InconsistentVersionWarning)
        else:
            warnings.simplefilter("ignore")
        return joblib.load(path)


def _verifikasi(model, bundle, metrics):
    """Pastikan model yang dimuat masih menghasilkan akurasi seperti seharusnya.

    Dipakai saat model terpaksa dimuat walau versi scikit-learn berbeda.
    Kalau akurasinya meleset, lebih baik berhenti daripada menyajikan
    prediksi yang salah tanpa disadari.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    df = pd.read_csv(DATASET_PATH, sep=";")
    fitur = bundle["features"]
    X = df[fitur].astype(float).values
    y = df["diagnosis"].astype(int).values
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    akurasi = accuracy_score(y_te, model.predict(X_te))
    diharapkan = metrics["accuracy"]
    if abs(akurasi - diharapkan) > 0.01:
        raise RuntimeError(
            f"Model yang dimuat menghasilkan akurasi {akurasi:.4f}, "
            f"seharusnya {diharapkan:.4f}. Jalankan 'python train_model.py' "
            f"di komputer ini, atau instal dependensi lewat "
            f"'pip install -r requirements.txt'."
        )
    print(f"[ok]   Verifikasi model lolos (akurasi {akurasi:.4f}).")


def muat_artefak():
    """Muat model, scaler, dan metrik - setahan mungkin terhadap beda lingkungan.

    Urutan penanganan:
      1. File lengkap dan versi scikit-learn cocok  -> langsung dipakai.
      2. Versi berbeda, train_model.py tersedia     -> latih ulang otomatis.
      3. Versi berbeda, train_model.py tidak ada    -> muat apa adanya, lalu
                                                       diverifikasi akurasinya.
    """
    path_model = os.path.join(MODEL_DIR, "rf_model.pkl")
    path_scaler = os.path.join(MODEL_DIR, "scaler.pkl")
    path_metrics = os.path.join(MODEL_DIR, "metrics.json")

    ada_semua = all(os.path.exists(p)
                    for p in (path_model, path_scaler, path_metrics))

    if ada_semua:
        try:
            model = _muat_pickle(path_model)
            bundle = _muat_pickle(path_scaler)
            with open(path_metrics, encoding="utf-8") as f:
                return model, bundle, json.load(f)
        except Exception as e:
            print(f"[info] Model dibuat oleh lingkungan berbeda: "
                  f"{type(e).__name__}: {str(e).splitlines()[0]}")
    else:
        print("[info] File model belum lengkap.")

    # Coba latih ulang - hanya bisa kalau train_model.py ikut disalin.
    try:
        import train_model
    except ImportError:
        train_model = None

    if train_model is not None:
        print("[info] Melatih ulang model di komputer ini ...")
        train_model.main()
        model = joblib.load(path_model)
        bundle = joblib.load(path_scaler)
        with open(path_metrics, encoding="utf-8") as f:
            metrics = json.load(f)
        print("[ok]   Model siap dipakai.")
        return model, bundle, metrics

    if not ada_semua:
        raise FileNotFoundError(
            "File model tidak ada dan train_model.py juga tidak ditemukan. "
            "Salin folder 'model/', 'dataset/', dan file 'train_model.py' "
            "ke folder project ini."
        )

    # Jalan terakhir: muat apa adanya, lalu buktikan modelnya masih benar.
    print("[warn] train_model.py tidak ditemukan - model dimuat apa adanya "
          "lalu diverifikasi.")
    print("[warn] Sebaiknya jalankan: pip install -r requirements.txt")
    model = _muat_pickle(path_model, ketat=False)
    bundle = _muat_pickle(path_scaler, ketat=False)
    with open(path_metrics, encoding="utf-8") as f:
        metrics = json.load(f)
    _verifikasi(model, bundle, metrics)
    return model, bundle, metrics


MODEL, BUNDLE, METRICS = muat_artefak()
FITUR = BUNDLE["features"]          # 8 nama fitur GWO
INDEKS = BUNDLE["indices"]          # posisi 8 fitur itu di antara 30 fitur
SCALER = BUNDLE["scaler"]
RENTANG = METRICS["feature_ranges"]


# --------------------------------------------------------------------------
# Inti prediksi
# --------------------------------------------------------------------------
def skalakan(nilai_asli: dict) -> np.ndarray:
    """Ubah 8 nilai klinis asli menjadi 8 nilai ternormalisasi 0-1.

    Scaler di-fit pada 30 fitur, sementara input hanya 8. Karena MinMaxScaler
    bekerja per kolom (kolom lain tidak saling memengaruhi), caranya: buat
    vektor 30 elemen, isi 8 posisi sesuai indeks GWO, transform, lalu ambil
    kembali hanya 8 posisi itu.
    """
    vektor = np.zeros((1, 30))
    for posisi, nama in zip(INDEKS, FITUR):
        vektor[0, posisi] = nilai_asli[nama]
    hasil = SCALER.transform(vektor)[:, INDEKS]
    return np.clip(hasil, 0.0, 1.0)     # jaga-jaga input di luar rentang WDBC


def _cek_input_ternormalisasi(nilai_asli: dict) -> bool:
    """Deteksi input yang sudah di-scale ke 0-1, bukan format raw model.

    Beberapa fitur klinis seperti area_mean, radius_se, perimeter_worst,
    smoothness_worst memang bisa bernilai < 1 bila data sudah dinormalisasi,
    tetapi range asli model untuk fitur-fitur itu jauh lebih besar. Jika nilai
    satu fitur yang seharusnya berada pada kisaran > 1 turun drastis ke bawah
    setengah min-nya, maka input kemungkinan sudah di-normalisasi.
    """
    for nama in (
        "area_mean",
        "radius_se",
        "compactness_mean",
        "perimeter_se",
        "perimeter_worst",
        "smoothness_worst",
        "concavity_worst",
        "fractal_dimension_worst",
    ):
        if nama not in nilai_asli:
            continue
        nilai = float(nilai_asli[nama])
        minimum = RENTANG[nama]["min"]
        if minimum > 1 and nilai < minimum * 0.5:
            return True
    return False


def prediksi(nilai_asli: dict) -> dict:
    if _cek_input_ternormalisasi(nilai_asli):
        raise ValueError(
            "Input terlihat sudah dinormalisasi (skala 0-1). "
            "Gunakan data klinis asli sesuai format training model."
        )

    X = skalakan(nilai_asli)
    kelas = int(MODEL.predict(X)[0])
    proba = MODEL.predict_proba(X)[0]
    p_ganas = float(proba[1])

    # Kontribusi tiap fitur = importance global x nilai ternormalisasi,
    # dinormalkan jadi persentase. Dipakai untuk bar "Feature Contribution".
    imp = np.array([METRICS["feature_importance"][n] for n in FITUR])
    kontrib = imp * X[0]
    kontrib = kontrib / kontrib.sum() * 100 if kontrib.sum() > 0 else imp * 100

    peringatan = [
        f"{n}: {nilai_asli[n]:g} di luar rentang data latih "
        f"({RENTANG[n]['min']:g} - {RENTANG[n]['max']:g})"
        for n in FITUR
        if not (RENTANG[n]["min"] <= nilai_asli[n] <= RENTANG[n]["max"])
    ]

    return {
        "ok": True,
        "kelas": kelas,
        "label": LABEL[kelas],
        "label_id": LABEL_ID[kelas],
        "keyakinan": round(float(proba[kelas]) * 100, 2),
        "prob_ganas": round(p_ganas * 100, 2),
        "prob_jinak": round(float(proba[0]) * 100, 2),
        "kontribusi": [
            {"fitur": n, "persen": round(float(v), 1),
             "nilai": nilai_asli[n], "ternormalisasi": round(float(x), 4)}
            for n, v, x in sorted(zip(FITUR, kontrib, X[0]),
                                  key=lambda t: -t[1])
        ],
        "peringatan": peringatan,
    }


def baca_input(sumber) -> dict:
    """Ambil 8 nilai dari request, validasi wajib-isi dan tipe angka."""
    nilai, kurang, salah = {}, [], []
    for nama in FITUR:
        mentah = sumber.get(nama)
        if mentah is None or str(mentah).strip() == "":
            kurang.append(nama)
            continue
        try:
            nilai[nama] = float(mentah)
        except (TypeError, ValueError):
            salah.append(nama)
    if kurang:
        raise ValueError("Field wajib diisi: " + ", ".join(kurang))
    if salah:
        raise ValueError("Nilai harus berupa angka: " + ", ".join(salah))
    return nilai


# --------------------------------------------------------------------------
# Route
# --------------------------------------------------------------------------
@app.route("/")
def landing():
    """Landing page."""
    return render_template("landing.html", metrics=METRICS)


@app.route("/diagnostics")
def diagnostics():
    """Halaman analisis: form 8 fitur GWO + panel hasil."""
    fields = [
        {
            "nama": n,
            "min": RENTANG[n]["min"],
            "max": RENTANG[n]["max"],
            "contoh": RENTANG[n]["contoh"],
        }
        for n in FITUR
    ]
    return render_template("index.html", fields=fields, metrics=METRICS)


@app.route("/batch-prediction")
def batch_prediction():
    """Halaman batch prediction dari file CSV."""
    return render_template("batch_prediction.html", metrics=METRICS)


@app.route("/batch-predict", methods=["POST"])
def route_batch_predict():
    """Prediksi multi-baris dari CSV upload."""
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"ok": False, "error": "Pilih file CSV terlebih dahulu."}), 400
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"ok": False, "error": "File harus berformat CSV."}), 400

    try:
        df = pd.read_csv(file)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"File CSV tidak valid: {exc}"}), 400

    missing = [kolom for kolom in FITUR if kolom not in df.columns]
    if missing:
        return jsonify({
            "ok": False,
            "error": "Kolom CSV tidak lengkap. Kolom yang dibutuhkan: " + ", ".join(FITUR),
        }), 400

    results = []
    for idx, row in df.iterrows():
        if any(pd.isna(row[nama]) for nama in FITUR):
            results.append({
                "row": idx + 2,
                "status": "invalid",
                "error": "Ada nilai kosong pada fitur yang diperlukan.",
            })
            continue

        payload = {nama: float(row[nama]) for nama in FITUR}
        if _cek_input_ternormalisasi(payload):
            results.append({
                "row": idx + 2,
                "status": "invalid",
                "error": (
                    "Input terlihat sudah dinormalisasi (skala 0-1). "
                    "Gunakan data klinis asli sesuai format training model."
                ),
            })
            continue

        try:
            pred = prediksi(payload)
        except Exception as exc:
            results.append({
                "row": idx + 2,
                "status": "invalid",
                "error": str(exc),
            })
            continue

        actual = row.get("diagnosis")
        actual_label = None
        if pd.notna(actual):
            try:
                actual_label = LABEL[int(actual)]
            except (TypeError, ValueError):
                actual_label = str(actual)

        results.append({
            "row": idx + 2,
            "status": "ok",
            "features": {nama: float(row[nama]) for nama in FITUR},
            "predicted_label": pred["label_id"],
            "predicted_class": pred["kelas"],
            "confidence": pred["keyakinan"],
            "prob_ganas": pred["prob_ganas"],
            "prob_jinak": pred["prob_jinak"],
            "actual_label": actual_label,
            "peringatan": pred["peringatan"],
        })

    summary = {
        "total": len(results),
        "benign": sum(1 for item in results if item.get("status") == "ok" and item.get("predicted_label") == "Jinak"),
        "malignant": sum(1 for item in results if item.get("status") == "ok" and item.get("predicted_label") == "Ganas"),
    }

    return jsonify({
        "ok": True,
        "results": results,
        "features": FITUR,
        "summary": summary,
    })


@app.route("/predict", methods=["POST"])
def route_predict():
    sumber = request.get_json(silent=True) or request.form
    try:
        nilai = baca_input(sumber)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    return jsonify(prediksi(nilai))


@app.route("/api/sample")
def route_sample():
    """Ambil satu contoh kasus nyata dari dataset, dikembalikan dalam skala ASLI.

    Dipakai tombol "Isi Contoh" agar demo cepat tanpa mengetik 8 angka.
    """
    tipe = request.args.get("type", "malignant").lower()
    target = 1 if tipe.startswith("m") else 0

    df = pd.read_csv(DATASET_PATH, sep=";")
    kandidat = df[df["diagnosis"] == target]
    baris = kandidat.sample(1).iloc[0]

    # Balik dari skala 0-1 ke nilai asli: nilai = min + x * (max - min)
    nilai = {
        n: round(RENTANG[n]["min"]
                 + float(baris[n]) * (RENTANG[n]["max"] - RENTANG[n]["min"]), 5)
        for n in FITUR
    }
    return jsonify({"ok": True, "nilai": nilai,
                    "label_sebenarnya": LABEL[target]})


@app.route("/api/metrics")
def route_metrics():
    return jsonify(METRICS)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5001)
