from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the pipeline from the saved file
pipeline = joblib.load('C:\\Study\\Projects\\CyberSecurity\\Flask\\phishing.pkl')

# Ensure that the loaded pipeline has the correct structure
if 'classifier' in pipeline.named_steps:
    best_model = pipeline.named_steps['classifier']
else:
    best_model = pipeline

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input URL from the form
    url = request.form['url']

    # Define the input data
    url_length = len(url)
    X = np.array([[url_length]])

    # Provide feature names to the input data
    feature_names = ['url_length']
    X = pd.DataFrame(X, columns=feature_names)

    # Predict the label of the input data
    predicted_label = best_model.predict(X)[0]

    # Map the predicted label to a user-friendly label
    if predicted_label == 0:
        predicted_label = "Safe Website"
    else:
        predicted_label = "Phishing Website (not safe)"

    # Create a dictionary to pass the predicted label to the JavaScript code
    result = {'predicted_label': predicted_label}

    # Return a JSON response
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)