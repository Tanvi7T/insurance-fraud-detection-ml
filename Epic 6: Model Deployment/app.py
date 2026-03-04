from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "Insurance Fraud Detection App Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    prediction = model.predict([data])
    return jsonify({"Fraud Prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
