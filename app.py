import streamlit as st
import pickle
import numpy as np
import os

# 1. Title and Description
st.title("Student Exam Predictor")
st.write("Enter study and sleep hours to predict if the student will Pass or Fail.")

# 2. Load the Model
model_path = 'SVM.pkl'

if os.path.exists(model_path):
    try:
        # We use st.cache_resource so we load the model only once
        @st.cache_resource
        def load_model():
            with open(model_path, 'rb') as file:
                return pickle.load(file)
        
        model = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model = None
else:
    st.error("File 'my_model.pkl' not found. Please upload it.")
    model = None

# 3. Create Input Fields
study_hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0, value=5.0, step=0.1)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.1)

# 4. Predict Button
if st.button("Predict Result"):
    if model is not None:
        # Prepare input as 2D array
        features = np.array([[study_hours, sleep_hours]])
        
        # Predict
        prediction = model.predict(features)
        result = prediction[0]
        
        # Display Result
        if result == "Pass":
            st.success(f"Prediction: {result}")
        else:
            st.error(f"Prediction: {result}")
    else:
        st.warning("Model is not loaded, cannot predict.")
