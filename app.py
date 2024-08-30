import streamlit as st
import pandas as pd
from components.model.model import model_try
from components.research.research import research
import joblib

st.title('Obesity and CVR Risk Prediction')
st.write('Author: Elmer Raúl Escandón Tufino')

@st.cache_data
def load_test_data():
    return pd.read_csv("./data/test.csv")

@st.cache_data
def load_model():
    return joblib.load("./models/GBT_ENC_model.pkl")

# Main app
def main():
    classification_model = load_model();

    Model, Research = st.tabs(["Model", "Research"])
    with Model:
        model_try(classification_model)
    with Research:
        research()

if __name__ == '__main__':
    main()