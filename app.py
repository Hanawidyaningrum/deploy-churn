from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# ── Load model terbaik (KNN, akurasi 78%) & preprocessing ────────────────────
model   = pickle.load(open('model_knn.pkl', 'rb'))
scaler  = pickle.load(open('scaler.pkl',    'rb'))
imputer = pickle.load(open('imputer.pkl',   'rb'))


@app.route('/')
def home():
    # Halaman form → index.html
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ── 1. Ambil input dari form ──────────────────────────────────────────
        raw = [
            int(request.form['SeniorCitizen']),
            float(request.form['tenure']),
            int(request.form['InternetService']),
            int(request.form['OnlineSecurity']),
            int(request.form['TechSupport']),
            int(request.form['Contract']),
            int(request.form['PaperlessBilling']),
            int(request.form['PaymentMethod']),
            float(request.form['MonthlyCharges']),
            float(request.form['TotalCharges']),
        ]

        data = np.array(raw, dtype=float).reshape(1, -1)

        # ── 2. Preprocessing ──────────────────────────────────────────────────
        data = imputer.transform(data)
        data = scaler.transform(data)

        # ── 3. Prediksi ───────────────────────────────────────────────────────
        pred       = model.predict(data)[0]
        prob       = model.predict_proba(data)[0]
        prob_churn = round(prob[1] * 100, 1)
        prob_stay  = round(prob[0] * 100, 1)

        if pred == 1:
            result = "CHURN — Pelanggan kemungkinan BERHENTI"
            warna  = "red"
        else:
            result = "TIDAK CHURN — Pelanggan tetap berlangganan"
            warna  = "green"

        inputs = {f: request.form[f] for f in request.form}

        # ── Hasil tampil di halaman TERPISAH → result.html ───────────────────
        return render_template(
            'result.html',
            prediction=result,
            warna=warna,
            prob_churn=prob_churn,
            prob_stay=prob_stay,
            inputs=inputs,
        )

    except KeyError as e:
        return render_template('result.html',
                               prediction=f"Field tidak ditemukan: {e}",
                               warna='gray', prob_churn=None,
                               prob_stay=None, inputs=None)

    except ValueError as e:
        return render_template('result.html',
                               prediction=f"Nilai tidak valid: {e}",
                               warna='gray', prob_churn=None,
                               prob_stay=None, inputs=None)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('result.html',
                               prediction=f"Error: {str(e)}",
                               warna='gray', prob_churn=None,
                               prob_stay=None, inputs=None)


if __name__ == '__main__':
    app.run(debug=True)