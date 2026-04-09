import streamlit as st
import pandas as pd
import time
import joblib

# ✅ Ingestion layer
from src.data_ingestion.mongo_client import insert_application, fetch_all_applications

# ✅ Load trained model
model = joblib.load("model.pkl")

# Theme
PRIMARY_COLOR = "#2E7D32"
ACCENT_COLOR = "#5D4037"


def main():
    st.set_page_config(page_title="EquiLend AI - Credit Scoring", layout="wide")

    st.title("⚖️ EquiLend AI: Transparent Credit Scoring")
    st.markdown("### AI-powered credit risk assessment using ML")

    # Sidebar
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

                # ✅ Fix 1: Age validation
                if age < 18:
                    st.error("Applicant must be at least 18 years old.")
                    return

                # Convert inputs
                gender_val = 0 if gender == "Male" else 1
                employment_val = int(employment.split()[0])

                # Prepare input
                input_data = pd.DataFrame([{
                    "gender": gender_val,
                    "monthly_income": income,
                    "utility_bill_average": utility_bill,
                    "repayment_history_pct": repayment_history,
                    "employment_length": employment_val
                }])

                # ✅ Fix 2 & 3: ML prediction
                prediction = model.predict_proba(input_data)[0][1]
                risk_level = "High" if prediction > 0.5 else "Low"

                # Output
                st.success(f"Analysis Complete for {name}")
                st.metric("Risk Probability", round(prediction, 2))
                st.write(f"Recommended Decision: **{risk_level} Risk**")

                # ✅ Fix 4: Save to MongoDB
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

    # ------------------ DASHBOARD ------------------
    elif choice == "Dashboard":
        st.subheader("📊 Applications Dashboard")

        data = fetch_all_applications()

        if data:
            df = pd.DataFrame(data)
            st.dataframe(df)

            st.markdown("### Risk Distribution")
            st.bar_chart(df["risk_level"].value_counts())
        else:
            st.warning("No applications found.")


# Run
if __name__ == '__main__':
    main()