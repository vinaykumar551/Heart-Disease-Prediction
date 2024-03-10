import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the scaler object
scaler = joblib.load('scaler.pkl')

# Load the trained models
model_names = ['NaiveBayes']

models = {}

for model_name in model_names:
    models[model_name] = joblib.load(f'{model_name}.pkl')

st.title('Heart Disease Health Indicators :anatomical_heart:')

st.divider()

st.header('Input Features')

# Define mappings for categorical features
diabetes_map = {0: 'No Diabetes', 1: 'Pre-diabetes', 2: 'Diabetes'}
fruits_map = {0: 'Does not consume fruits', 1: 'Consumes fruits regularly'}
gen_hlth_map = {1: 'Excellent', 2: 'Good', 3: 'Average', 4: 'Fair', 5: 'Poor'}
pred_map = {0: 'No risk of Heart Disease found', 1: 'Having s high risk of Heart Disease'}

# Display input features with text descriptions
high_bp = st.radio('High Blood Pressure', ['No', 'Yes'])
high_chol = st.radio('High Cholesterol', ['No', 'Yes'])
chol_check = st.radio('Cholesterol Checked', ['No', 'Yes'])
bmi = st.slider('BMI', min_value=12, max_value=98, step=1, value=50)
stroke = st.radio('Stroke', ['No', 'Yes'])
diabetes = st.selectbox('Diabetes', options=list(diabetes_map.values()))
fruits = st.radio('Fruits', options=list(fruits_map.values()))
gen_hlth = st.select_slider('General Health', options=list(gen_hlth_map.values()).__reversed__())

# Predict
if st.button('Predict'):
    high_bp_numeric = 1 if high_bp == 'Yes' else 0
    high_chol_numeric = 1 if high_chol == 'Yes' else 0
    chol_check_numeric = 1 if chol_check == 'Yes' else 0
    diabetes_numeric = next(key for key, value in diabetes_map.items() if value == diabetes)
    stroke_numeric = 1 if stroke == 'Yes' else 0
    fruits_numeric = next(key for key, value in fruits_map.items() if value == fruits)
    gen_hlth_numeric = next(key for key, value in gen_hlth_map.items() if value == gen_hlth)
    
    input_data = np.array([high_bp_numeric, high_chol_numeric, chol_check_numeric, bmi, stroke_numeric, diabetes_numeric, fruits_numeric, gen_hlth_numeric])
    
    scaled_input_data = scaler.transform(input_data.reshape(1,-1))

    st.balloons()

    st.divider()

    for selected_model_name, selected_model in models.items():
        prediction = selected_model.predict(scaled_input_data)

        st.subheader(f'Prediction using {selected_model_name}')

        pred_str = ", ".join([pred_map[pred] for pred in prediction])
        if 0 in prediction:
            st.success(pred_str)
        else:
            st.error(pred_str)
