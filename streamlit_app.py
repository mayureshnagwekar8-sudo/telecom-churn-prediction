"""Streamlit app for telecom customer churn prediction.

Run `python train.py` once to create churn_model.joblib, then:
    streamlit run streamlit_app.py
"""

import joblib
import pandas as pd
import streamlit as st

from churn import encode

MODEL_PATH = "churn_model.joblib"


@st.cache_resource
def load_bundle(path: str = MODEL_PATH):
    return joblib.load(path)


def predict(bundle, user_input: dict) -> tuple[int, float]:
    X = encode(pd.DataFrame([user_input]), columns=bundle["columns"])
    pipeline = bundle["pipeline"]
    return int(pipeline.predict(X)[0]), float(pipeline.predict_proba(X)[0][1])


def main():
    st.set_page_config(page_title="Churn Prediction", layout="centered")
    st.title("Telecom Customer Churn Prediction")
    st.markdown("Enter a customer's details to estimate whether they will churn.")

    bundle = load_bundle()

    with st.form("input_form"):
        user_input = {
            "tenure": st.slider("Tenure (months)", min_value=0, max_value=72, value=12),
            "SeniorCitizen": st.selectbox("Senior citizen", [0, 1], format_func=lambda v: "Yes" if v else "No"),
        }
        for col, options in bundle["options"].items():
            user_input[col] = st.selectbox(col, options)
        for col, (lo, hi, default) in bundle["numeric_ranges"].items():
            user_input[col] = st.number_input(col, min_value=lo, max_value=hi, value=default)

        submitted = st.form_submit_button("Predict")

    if submitted:
        churn, probability = predict(bundle, user_input)
        st.subheader("Prediction")
        if churn:
            st.error(f"Likely to churn (probability {probability:.0%})")
        else:
            st.success(f"Likely to stay (churn probability {probability:.0%})")

        with st.expander("Model input features"):
            X = encode(pd.DataFrame([user_input]), columns=bundle["columns"])
            st.dataframe(X.T.rename(columns={0: "value"}))


if __name__ == "__main__":
    main()
