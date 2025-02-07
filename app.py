from flask import Flask, request, jsonify
import numpy as np
from joblib import load
import pandas as pd
from flask_cors import CORS

# Load the saved model
model = load('isolation_forest_model.joblib')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json(force=True)
        
        # Convert JSON to DataFrame
        input_data = pd.DataFrame([{
            "temperature": data["temperature"],
            "pressure": data["pressure"],
            "vibration": data["vibration"],
            "humidity": data["humidity"]
        }])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Convert prediction to binary (0 for normal, 1 for anomaly)
        prediction = np.where(prediction == 1, 0, 1)
        
        # Return the prediction as JSON
        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)})


# Run the Flask app
app.run()