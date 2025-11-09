import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, clean_replace_zero_with_median, FEATURES, TARGET

@st.cache_data
def load_model(model_path='model.pkl', scaler_path='scaler.pkl'):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

st.set_page_config(page_title='Diabetes Prediction Dashboard', layout='wide')

st.title('Diabetes Prediction Dashboard')

st.sidebar.header('Patient Input')
mode = st.sidebar.selectbox('Mode', ['Manual input', 'Use sample from dataset'])

df_raw = load_data('data/diabetes.csv')
df = clean_replace_zero_with_median(df_raw)

if mode == 'Use sample from dataset':
    idx = st.sidebar.number_input('Sample index', min_value=0, max_value=len(df)-1, value=0)
    sample = df.iloc[idx]
    st.sidebar.write(sample[FEATURES])
else:
    pregnancies = st.sidebar.number_input('Pregnancies', 0, 20, 1)
    glucose = st.sidebar.number_input('Glucose', 0, 300, 120)
    bp = st.sidebar.number_input('BloodPressure', 0, 150, 70)
    skin = st.sidebar.number_input('SkinThickness', 0, 100, 20)
    insulin = st.sidebar.number_input('Insulin', 0.0, 900.0, 79.0)
    bmi = st.sidebar.number_input('BMI', 0.0, 70.0, 32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.471)
    age = st.sidebar.number_input('Age', 1, 120, 33)

model, scaler = load_model('model.pkl', 'scaler.pkl')

st.header('Exploratory Data Analysis')
col1, col2 = st.columns([2,1])
with col1:
    st.subheader('Outcome distribution')
    fig, ax = plt.subplots()
    sns.countplot(x='Outcome', data=df, ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader('Age distribution')
    st.write(df['Age'].describe())
    fig2, ax2 = plt.subplots()
    sns.histplot(df['Age'], bins=20, ax=ax2)
    st.pyplot(fig2)

st.header('Prediction')

if mode == 'Use sample from dataset':
    X = sample[FEATURES].values.reshape(1, -1)
else:
    X = np.array([pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]).reshape(1, -1)

X_s = scaler.transform(X)
prob = model.predict_proba(X_s)[0][1]
pred = model.predict(X_s)[0]

st.metric(label='Predicted probability of diabetes', value=f"{prob*100:.2f}%")
st.write('Predicted class:', 'Positive (Diabetes)' if pred==1 else 'Negative (No Diabetes)')
st.write('---')
st.subheader('Patient features')
st.table(pd.DataFrame(X.reshape(1,-1), columns=FEATURES))
st.info('To retrain the model or update dataset, run `python model_training.py` then restart the app.')
