import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
#from flask_cors import CORS
#from gevent.pywsgi import WSGIServer

# ---------------- APP SETUP ----------------
app2 = Flask(__name__)
#CORS(app2)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("PlantDNet.h5", compile=False)
print("✅ Model loaded successfully")

# ---------------- CLASS LABELS ----------------
disease_classes = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# ---------------- DISEASE INFO ----------------
disease_info = {
    "Pepper__bell___Bacterial_spot": {
        "description": "Bacterial spot causes dark water-soaked lesions on leaves.",
        "treatment": "Apply copper-based bactericides and avoid overhead watering."
    },
    "Pepper__bell___healthy": {
        "description": "The plant is healthy with no visible disease symptoms.",
        "treatment": "No treatment needed. Maintain proper care."
    },
    "Potato___Early_blight": {
        "description": "Early blight causes brown circular spots on leaves.",
        "treatment": "Use fungicides like mancozeb and remove infected leaves."
    },
    "Potato___Late_blight": {
        "description": "Late blight causes rapid leaf decay and plant collapse.",
        "treatment": "Apply systemic fungicides immediately."
    },
    "Potato___healthy": {
        "description": "Healthy potato plant.",
        "treatment": "No treatment required."
    },
    "Tomato_Bacterial_spot": {
        "description": "Bacterial spot affects tomato leaves and fruits.",
        "treatment": "Use copper sprays and certified seeds."
    },
    "Tomato_Early_blight": {
        "description": "Early blight causes leaf yellowing and brown lesions.",
        "treatment": "Apply fungicide and rotate crops."
    },
    "Tomato_Late_blight": {
        "description": "Late blight causes fast-spreading leaf damage.",
        "treatment": "Destroy infected plants and apply fungicides."
    },
    "Tomato_healthy": {
        "description": "The tomato plant is healthy.",
        "treatment": "No treatment required."
    }
}

# ---------------- IMAGE PREDICTION ----------------
def model_predict(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    preds = model.predict(img_array)
    return preds[0]

# ---------------- PREDICT API ----------------
@app2.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file received"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    preds = model_predict(file_path)
    index = int(np.argmax(preds))

    disease_name = disease_classes[index]
    confidence = float(preds[index])

    info = disease_info.get(
        disease_name,
        {
            "description": "Disease information not available.",
            "treatment": "Consult an agriculture expert."
        }
    )

    return jsonify({
        "prediction": disease_name,
        "confidence": confidence,
        "description": info["description"],
        "treatment": info["treatment"]
    }), 200

# ---------------- HEALTH CHECK ----------------
@app2.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "API running"}), 200

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    print("🚀 API running at http://0.0.0.0:5000")
    #http_server = WSGIServer(("0.0.0.0", 5000), app2)
    #http_server.serve_forever()
    app2.run(host="0.0.0.0", port=10000)
