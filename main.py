from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

MODEL_PATH = 'model_penyakit_daun_kacang_fixed.h5'

app = Flask(__name__)

# Muat model sekali saat server dinyalakan
try:
    model = load_model(MODEL_PATH)
    print(f"✓ Model berhasil dimuat dari {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"✗ Gagal memuat model dari {MODEL_PATH}: {e}")


def preprocess_image(image_file, target_size=(150, 150)):
    """Buka gambar, resize ke 150x150, konversi ke array, normalisasi, dan tambahkan batch dimension."""
    img = Image.open(image_file).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'status': 'error', 'pesan': 'Model belum dimuat. Pastikan file model_penyakit_daun_kacang.h5 ada.'}), 500

    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'status': 'error', 'pesan': 'Tidak ada gambar yang diterima'}), 400

    file = request.files['file']

    try:
        processed_image = preprocess_image(file)
        prediction = model.predict(processed_image, verbose=0)

        # Asumsi output model biner
        raw_score = float(prediction[0][0])
        
        # Hitung confidence untuk setiap kelas
        if raw_score < 0.5:
            label = 'Sehat'
            confidence = (1 - raw_score) * 100
        else:
            label = 'Tidak Sehat'
            confidence = raw_score * 100
        
        confidence = round(confidence, 2)

        return jsonify({
            'status': 'sukses', 
            'hasil_prediksi': label,
            'confidence': confidence,
            'raw_score': round(raw_score, 4)
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'pesan': f'Gagal memproses gambar: {str(e)}'}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Sistem Deteksi Penyakit Daun Kacang - Server Running")
    print("="*60)
    print("Buka browser di: http://localhost:5000")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
