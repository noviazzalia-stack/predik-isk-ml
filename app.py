from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# =========================
# LOAD MODEL & KOLOM
# =========================
try:
    model = joblib.load("model_isk.pkl")
    training_columns = joblib.load("columns.pkl")
    print("✅ Model & kolom berhasil dimuat")
except Exception as e:
    print("❌ ERROR load model:", e)
    model = None
    training_columns = None


# =========================
# PREPROCESS INPUT
# =========================
def preprocess_input(data):
    df = pd.DataFrame([data])

    # one-hot encoding
    df = pd.get_dummies(df)

    # samakan kolom
    df = df.reindex(columns=training_columns, fill_value=0)

    return df


# =========================
# ROUTE TEST
# =========================
@app.route("/")
def home():
    return "🚀 API ISK aktif di Railway!"


# =========================
# ROUTE PREDIKSI
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({
                "status": "error",
                "message": "Model belum ter-load"
            })

        data = request.get_json()

        if not data:
            return jsonify({
                "status": "error",
                "message": "Data kosong"
            })

        # preprocess
        df = preprocess_input(data)

        # prediksi
        prediction = model.predict(df)[0]
        proba = model.predict_proba(df)[0]

        # mapping label
        label_map = {
            0: "Rendah",
            1: "Sedang",
            2: "Tinggi"
        }

        hasil = label_map.get(int(prediction), "Tidak diketahui")

        return jsonify({
            "status": "success",
            "prediksi": hasil,
            "probabilitas": {
                "rendah": float(proba[0]),
                "sedang": float(proba[1]),
                "tinggi": float(proba[2])
            }
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })


# =========================
# RUN SERVER (WAJIB UNTUK RAILWAY)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)