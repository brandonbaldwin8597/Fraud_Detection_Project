import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


st.sidebar.title('Fraud Prediction')
html_temp = """
<div style="background-color:blue;padding:10px">
<h2 style="color:white;text-align:center;">Fraud Prediction Model</h2>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)


V14 = st.sidebar.slider("V14:", -19.214, 10.527, step=0.05)
V10 = st.sidebar.slider("V10:", -24.588, 23.745, step=0.05)
V4 = st.sidebar.slider("V14:", -5.600, 16.875, step=0.05)
V12 = st.sidebar.slider("V12:", -18.683, 7.848, step=0.05)
V11 = st.sidebar.slider("V11:", -4.797, 12.018, step=0.05)

model = pickle.load(open("randomforest_model","rb"))



my_dict = {
    "V14": V14,
    "V10": V10,
    "V4": V4,
    "V12": V12,
    "V11": V11
}

df = pd.DataFrame.from_dict([my_dict])


st.header("Credit Card Fraud Prediction")
st.table(df)

columns = ['V14',
           'V10',
           'V4',
           'V12',
           'V11']


st.subheader("Edit Variable Settings")

if st.button("Predict"):
    prediction = model.predict(df)
    value = int(prediction[0])
    if value == 0:
        value = "Safe from fraud"
    else:
        value = "FRAUD DETECTED"
    st.success("Fraud detection status: {}.".format(value))