import numpy as np
from flask import Flask, request, render_template
import pickle
import os

# 1. Initialize App
app = Flask(__name__)

# 2. Load Model Safely
model_path = 'SVM.pkl'

if not os.path.exists(model_path):
    print("ERROR: 'my_model.pkl' not found! Please run your training script first.")
    model = None
else:
    try:
        model = pickle.load(open(model_path, 'rb'))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        model = None

# 3. Home Route
@app.route('/')
def home():
    return render_template('index.html')

# 4. Predict Route
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="Error: Model is not loaded.")

    try:
        # Get data from form
        study = request.form.get('StudyHours')
        sleep = request.form.get('SleepHours')

        # Convert to float
        val1 = float(study)
        val2 = float(sleep)

        # Prepare for model (Reshape is important!)
        # The model expects [[val1, val2]], not [val1, val2]
        features = np.array([[val1, val2]])

        # Predict
        prediction = model.predict(features)
        output = prediction[0]

        return render_template('index.html', prediction_text=f'Result: {output}')

    except Exception as e:
        # This will print the specific error to your terminal and the webpage
        print(f"Prediction Error: {e}")
        return render_template('index.html', prediction_text=f'Error occurred: {e}')

if __name__ == "__main__":
    app.run(debug=True, port=5000)
