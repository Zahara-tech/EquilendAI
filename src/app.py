# src/app.py
import sys
import os
import streamlit as st
import pandas as pd
import time
import joblib
import shap
import matplotlib.pyplot as plt

# ------------------- FIX IMPORT PATH -------------------
# Add parent folder to sys.path so 'src' package can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ✅ Ingestion layer
from src.data_ingestion.mongo_client import insert_application, fetch_all_applications

# ------------------- LOAD MODEL -------------------
model = joblib.load("model.pkl")

# Theme colors
PRIMARY_COLOR = "#2E7D32"
ACCENT_COLOR = "#5D4037"

# ------------------- STREAMLIT APP -------------------
def main():
    st.set_page_config(page_title="EquiLend AI - Credit Scoring", layout="wide")
    st.title("⚖️ EquiLend AI: Transparent Credit Scoring")
    st.markdown("### AI-powered credit risk assessment using ML with explainability")

    # Sidebar navigation
    menu = ["New Application", "Dashboard"]
    choice = st.sidebar.selectbox("Navigation", menu)

    # ------------------ NEW APPLICATION ------------------
    if choice == "New Application":
        st.subheader("📄 Manual Loan Application")
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Full Name")
            age = st.number_input("Age", min_value=0, max_value=120)
            income = st.number_input("Monthly Income (₹)", min_value=0)
            gender = st.selectbox("Gender", ["Male", "Female"])
            employment = st.selectbox(
                "Employment Length",
                ["1 year", "2 years", "3 years", "5 years", "10 years"]
            )

        with col2:
            utility_bill = st.number_input("Average Utility Bill (₹)", min_value=0)
            repayment_history = st.slider("Repayment Consistency (%)", 0, 100, 50)

        if st.button("Analyze Risk"):
            with st.spinner("AI Model Calculating..."):
                time.sleep(1)

                # Input validation
                if age < 18:
                    st.error("Applicant must be at least 18 years old.")
                    return

                # Convert inputs to numeric
                gender_val = 0 if gender == "Male" else 1
                employment_val = int(employment.split()[0])

                # Prepare input for ML model
                input_data = pd.DataFrame([{
                    "gender": gender_val,
                    "monthly_income": income,
                    "utility_bill_average": utility_bill,
                    "repayment_history_pct": repayment_history,
                    "employment_length": employment_val
                }])

                # ML prediction
                prediction = model.predict_proba(input_data)[0][1]
                risk_level = "High" if prediction > 0.5 else "Low"

                # Display results
                st.success(f"Analysis Complete for {name}")
                st.metric("Risk Probability", round(prediction, 2))
                st.write(f"Recommended Decision: **{risk_level} Risk**")

                # ------------------- SHAP EXPLAINER -------------------
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer(input_data)

                    st.subheader("🟢 Feature Contribution (SHAP)")
                    # Bar plot of feature importance
                    fig = plt.figure()
                    shap.plots.bar(shap_values[0], max_display=10, show=False)
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"SHAP explanation failed: {e}")

                # ------------------- SAVE TO MONGO -------------------
                try:
                    insert_application({
                        "name": name,
                        "age": age,
                        "income": income,
                        "gender": gender,
                        "employment_length": employment_val,
                        "utility_bill": utility_bill,
                        "repayment_history": repayment_history,
                        "risk_probability": float(prediction),
                        "risk_level": risk_level
                    })
                    st.info("Application saved successfully!")
                except Exception as e:
                    st.error(f"Failed to save application: {e}")

    # ------------------ DASHBOARD ------------------
    elif choice == "Dashboard":
        st.subheader("📊 Applications Dashboard")
        try:
            data = fetch_all_applications()
            if data:
                df = pd.DataFrame(data)
                st.dataframe(df)
                st.markdown("### Risk Distribution")
                st.bar_chart(df["risk_level"].value_counts())
            else:
                st.warning("No applications found.")
        except Exception as e:
            st.error(f"Failed to fetch applications: {e}")


# ------------------- RUN APP -------------------
if __name__ == '__main__':
    main()