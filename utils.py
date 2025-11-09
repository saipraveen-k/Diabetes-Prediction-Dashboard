import pandas as pd
import numpy as np

FEATURES = [
    'Pregnancies','Glucose','BloodPressure','SkinThickness',
    'Insulin','BMI','DiabetesPedigreeFunction','Age'
]

TARGET = 'Outcome'

def load_data(path='data/diabetes.csv'):
    df = pd.read_csv(path)
    return df

def clean_replace_zero_with_median(df, cols=None):
    if cols is None:
        cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    df = df.copy()
    for c in cols:
        median = df.loc[df[c] > 0, c].median()
        df.loc[df[c] == 0, c] = median
    return df
