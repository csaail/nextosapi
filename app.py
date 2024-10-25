from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load your model and encoders here
model = joblib.load("os_recommendation_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get JSON data from the request
    input_data = pd.DataFrame([data])

    # Encode the data
    for column in input_data.columns:
        input_data[column] = label_encoders[column].transform(input_data[column])

    # Predict
    prediction = model.predict(input_data)
    os_prediction = target_encoder.inverse_transform(prediction)[0]
    
    reason = (
        "Windows is better suited for users who prioritize high malware protection, frequent updates, and strong multi-factor authentication support."
        if os_prediction == "Windows"
        else "Linux is ideal for users with high privacy concerns, requiring data encryption, and more control over user permissions."
    )

    return jsonify({"recommended_os": os_prediction, "reason": reason})

if __name__ == '__main__':
    app.run(debug=True)
