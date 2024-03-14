import streamlit as st
import numpy as np
import joblib

# Load the scaler object
scaler = joblib.load('scaler.pkl')

# Load the trained models
model_names = ['NaiveBayes', 'XGBoost']

st.title('Heart Disease Health Indicators :anatomical_heart:')
st.divider()

selected_model_name = st.selectbox('Select Model', model_names)
model = joblib.load(f'{selected_model_name}.pkl')
st.divider()

st.header('Input Features')

# Define mappings for categorical features
diabetes_map = {0: 'No Diabetes', 1: 'Pre-diabetes', 2: 'Diabetes'}
gen_hlth_map = {1: 'Excellent', 2: 'Good', 3: 'Average', 4: 'Fair', 5: 'Poor'}
pred_map = {0: 'No risk of Heart Disease found', 1: 'Having s high risk of Heart Disease'}

# Input features
high_bp = st.radio('High Blood Pressure', ['No', 'Yes'])
high_chol = st.radio('High Cholesterol', ['No', 'Yes'])
smoker = st.radio('Smoker', ['No', 'Yes'])
stroke = st.radio('Stroke', ['No', 'Yes'])
diabetes = st.selectbox('Diabetes', options=list(diabetes_map.values()))
gen_hlth = st.select_slider('General Health', options=list(gen_hlth_map.values()).__reversed__())
phys_hlth = st.slider('Physical Health', min_value=0, max_value=30, step=1)
diff_walk = st.radio('Difficulty Walking', ['No', 'Yes'])
age = st.number_input('Age', min_value=1, max_value=13, step=1)

def predict_input_data(model, input_data, scaler):
    scaled_input_data = scaler.transform(input_data.reshape(1, -1))
    prediction = model.predict(scaled_input_data)
    return prediction

# Predict
predict_btn_clicked = st.button('Predict')

if predict_btn_clicked:
    high_bp_numeric = 1 if high_bp == 'Yes' else 0
    high_chol_numeric = 1 if high_chol == 'Yes' else 0
    smoker_numeric = 1 if smoker == 'Yes' else 0
    stroke_numeric = 1 if stroke == 'Yes' else 0
    diabetes_numeric = next(key for key, value in diabetes_map.items() if value == diabetes)
    gen_hlth_numeric = next(key for key, value in gen_hlth_map.items() if value == gen_hlth)
    diff_walk_numeric = 1 if diff_walk == 'Yes' else 0

    input_data = np.array([high_bp_numeric, high_chol_numeric, smoker_numeric, stroke_numeric, diabetes_numeric, gen_hlth_numeric, phys_hlth, diff_walk_numeric, age])
    
    prediction = predict_input_data(model, input_data, scaler)
    st.balloons()
    st.divider()

    st.subheader(f'Prediction using {selected_model_name}')
    pred_str = pred_map[prediction[0]]
    if prediction[0] == 0:
        st.success(pred_str)
    else:
        st.error(pred_str)
