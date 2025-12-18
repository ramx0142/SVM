import numpy as np
from flask import Flask, request, render_template
import pickle

# 1. Initialize the App
app = Flask(__name__)

# 2. Load the trained model
model = pickle.load(open('SVM.pkl', 'rb'))

# 3. Define the Home Page (The Input Form)
@app.route('/')
def home():
    return render_template('index.html')

# 4. Define the Predict Logic
@app.route('/predict', methods=['POST'])
def predict():
    # Get values from the HTML form
    study_hours = float(request.form['StudyHours'])
    sleep_hours = float(request.form['SleepHours'])
    
    # Arrange them in a list like [[8.5, 9.0]]
    features = [np.array([study_hours, sleep_hours])]
    
    # Make Prediction
    prediction = model.predict(features)
    output = prediction[0]

    # Send result back to HTML
    return render_template('index.html', prediction_text=f'Student will: {output}')

# 5. Run the App
if __name__ == "__main__":
    app.run(debug=True)
