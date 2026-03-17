from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

print("Loading AI model...")
model = tf.keras.models.load_model("model.h5")
print("Model loaded successfully!")

class_names = [
    "Apple Scab",
    "Apple Black Rot",
    "Apple Cedar Rust",
    "Apple Healthy",
    "Blueberry Healthy",
    "Cherry Powdery Mildew",
    "Cherry Healthy",
    "Corn Gray Leaf Spot",
    "Corn Common Rust",
    "Corn Northern Leaf Blight",
    "Corn Healthy",
    "Grape Black Rot",
    "Grape Esca",
    "Grape Leaf Blight",
    "Grape Healthy",
    "Orange Huanglongbing",
    "Peach Bacterial Spot",
    "Peach Healthy",
    "Pepper Bell Bacterial Spot",
    "Pepper Bell Healthy",
    "Potato Early Blight",
    "Potato Late Blight",
    "Potato Healthy",
    "Raspberry Healthy",
    "Soybean Healthy",
    "Squash Powdery Mildew",
    "Strawberry Leaf Scorch",
    "Strawberry Healthy",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Tomato Healthy"
]

remedies = {
    "Apple Scab": "Spray fungicide and remove infected leaves",
    "Corn Northern Leaf Blight": "Use resistant varieties and rotate crops",
    "Potato Early Blight": "Avoid wet leaves and apply fungicide",
    "Tomato Leaf Mold": "Improve air circulation and remove infected leaves",
    "Tomato Late Blight": "Use fungicide and avoid excessive moisture",
    "Pepper Bell Bacterial Spot": "Use clean seeds and copper spray",
    "Strawberry Leaf Scorch": "Remove infected leaves and avoid water stress"
}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']

    img = Image.open(file).resize((224,224))
    img = np.array(img) / 255.0
    img = img.reshape(1,224,224,3)

    prediction = model.predict(img)

    index = int(np.argmax(prediction))
    confidence = round(float(np.max(prediction)) * 100, 2)

    disease_name = class_names[index]
    remedy = remedies.get(disease_name, "Consult agriculture expert for treatment")

    return jsonify({
        "disease": disease_name,
        "confidence": str(confidence) + "%",
        "remedy": remedy
    })

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True)

