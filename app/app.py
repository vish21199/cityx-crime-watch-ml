import streamlit as st
import joblib
from PyPDF2 import PdfReader

st.set_page_config(page_title="CityX Crime Watch", layout="wide")

# Load ML components
vectorizer = joblib.load("app/vectorizer.pkl")
model = joblib.load("app/model.pkl")
label_encoder = joblib.load("app/label_encoder.pkl")

# Upload PDF
uploaded_pdf = st.file_uploader("📄 Upload a police report PDF", type=["pdf"])

if uploaded_pdf:
    st.success("✅ File received.")

    # Extract text from PDF
    def extract_text_from_pdf(uploaded_file):
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text

    extracted_text = extract_text_from_pdf(uploaded_pdf)
    st.text_area("📄 Extracted Text:", extracted_text[:1000], height=200)

    # Predict crime category
    input_vec = vectorizer.transform([extracted_text])
    prediction_index = model.predict(input_vec)[0]
    prediction = label_encoder.inverse_transform([prediction_index])[0]
    st.success(f"🔍 Predicted Crime Category: **{prediction}**")

    # Assign severity
    severity_map = {
        "NON-CRIMINAL": 1, "SUSPICIOUS OCC": 1, "MISSING PERSON": 1, "RUNAWAY": 1, "RECOVERED VEHICLE": 1,
        "WARRANTS": 2, "OTHER OFFENSES": 2, "VANDALISM": 2, "TRESPASS": 2, "DISORDERLY CONDUCT": 2, "BAD CHECKS": 2,
        "LARCENY/THEFT": 3, "VEHICLE THEFT": 3, "FORGERY/COUNTERFEITING": 3, "DRUG/NARCOTIC": 3,
        "STOLEN PROPERTY": 3, "FRAUD": 3, "BRIBERY": 3, "EMBEZZLEMENT": 3,
        "ROBBERY": 4, "WEAPON LAWS": 4, "BURGLARY": 4, "EXTORTION": 4,
        "KIDNAPPING": 5, "ARSON": 5
    }

    severity = severity_map.get(prediction, "Unknown")
    st.info(f"🚨 Severity Level: **{severity}**")
