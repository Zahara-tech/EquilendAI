The conflict you've provided is quite messy because it contains nested merge markers and duplicated sections from the previous conflict resolution. I have cleaned up the `src/app.py` file to ensure:
1.  **Imports are clean**: All necessary libraries (`altair`, `shap`, `matplotlib`, etc.) are included.
2.  **Logic is unified**: The interactive **Altair** charts from the `main` branch are prioritized, but the **SHAP explainer** and **Bug fixes** from your branch are kept.
3.  **Threshold Artifacts**: The logic for handling `threshold_info` (which was a major source of the conflict) is now consistent throughout the file.

### Resolved `src/app.py`

```python
"""
EquiLend AI — Streamlit Dashboard
Fixes applied (Task 00):
  Bug 1 — Division-by-zero: utility_bill clamped to min 1 before any division
  Bug 2 — Age guard: hard block on applicants < 18
  Bug 3 — Linear formula replaced with trained XGBoost model
  Bug 4 — State persistence: decisions saved to MongoDB (session log fallback)

New feature: Threshold Optimizer (Lender Rules Engine)
"""

import os
import sys
import time
import shap
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")  # Must come before any other matplotlib import
import matplotlib.pyplot as plt

# ── Path setup ────────────────────────────────────────────────────────────────
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
for _p in (_SRC_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

# ── Constants ─────────────────────────────────────────────────────────────────
PRIMARY  = "#2E7D32"   # Forest Green
ACCENT   = "#5D4037"   # Soil Brown
ORANGE   = "#F57C00"

MODELS_DIR = os.path.join(_ROOT_DIR, "models")

# Support both data locations
_DATA_CANDIDATES = [
    os.path.join(_ROOT_DIR, "data", "equilend_mock_data.csv"),
    os.path.join(_ROOT_DIR, "scripts", "data", "equilend_mock_data.csv"),
]
DATA_PATH = next((p for p in _DATA_CANDIDATES if os.path.exists(p)), _DATA_CANDIDATES[0])

# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_artifacts():
    """Load saved model + preprocessor + test predictions + threshold info."""
    try:
        from models.train_xgb import load_artifacts
        return load_artifacts(MODELS_DIR)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def _sweep_thresholds(y_test_tup, y_prob_tup):
    """Threshold sweep — result cached so plots reuse it without recomputing."""
    from evaluation.thresholds import sweep_thresholds
    return sweep_thresholds(np.array(y_test_tup), np.array(y_prob_tup))

# ── Shared training helper ────────────────────────────────────────────────────

def _do_train():
    """Train XGBoost, clear cache, rerun. Returns AUC and business threshold info."""
    from models.train_xgb import train_and_save
    model, pre, y_test, y_prob, auc, threshold_info = train_and_save(DATA_PATH, MODELS_DIR)
    _load_artifacts.clear()
    return auc, threshold_info

# ── Page: New Application ─────────────────────────────────────────────────────

def page_new_application():
    st.subheader("Manual Loan Application")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Full Name")
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
        income = st.number_input("Monthly Income (₹)", min_value=0, step=500)
        gender = st.selectbox("Gender", ["Male", "Female", "Non-Binary"])

    with col2:
        utility_bill = st.number_input("Average Utility Bill (₹)", min_value=0, step=100)
        repayment_history = st.slider("Past Repayment Consistency (%)", 0, 100, 50)
        employment_length = st.selectbox(
            "Employment Length",
            ["< 1 year", "1-3 years", "4-7 years", "8+ years"],
        )

    if st.button("Analyze Risk", type="primary"):
        # Bug 2 fix: Age guard
        if age < 18:
            st.error("❌ Applicant must be at least 18 years old to apply.")
            return

        if not name.strip():
            st.warning("Please enter the applicant's full name.")
            return

        with st.spinner("AI Model Calculating…"):
            time.sleep(0.4)
            artifacts = _load_artifacts()

            if artifacts is None:
                # Bug 1 & 3 fallback
                safe_bill = max(utility_bill, 1)
                base_score = (income / safe_bill) * (repayment_history / 100)
                risk_level = "High" if base_score < 5 else "Low"
                st.warning("⚠️ ML model not trained yet. Using formula fallback.")
                st.metric("Formula Score", round(base_score, 2))
                st.write(f"Recommended Decision: **{risk_level} Risk**")
                return

            model, preprocessor, _, _, threshold_info = artifacts

            # Bug 3 fix: use the trained ML model
            input_df = pd.DataFrame([{
                "monthly_income": income,
                "utility_bill_average": max(utility_bill, 1), # Bug 1 fix
                "repayment_history_pct": repayment_history,
                "employment_length": employment_length,
                "gender": gender,
            }])
            X_proc = preprocessor.transform(input_df)
            prob_default = float(model.predict_proba(X_proc)[0, 1])

            threshold = st.session_state.get(
                "thresh_slider",
                float(threshold_info.get("threshold", 0.50)),
            )
            decision = "Deny — Default Risk" if prob_default >= threshold else "Approve"
            is_deny = prob_default >= threshold

        st.success(f"Analysis complete for **{name}**")

        m1, m2, m3 = st.columns(3)
        m1.metric("Default Probability", f"{prob_default * 100:.1f}%")
        m2.metric("Active Threshold", f"{threshold:.2f}")
        m3.metric("Decision", decision)

        if is_deny:
            st.error(f"🚫 {decision}")
        else:
            st.success(f"✅ {decision}")

        # SHAP EXPLAINER
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_proc)
            st.subheader("🟢 Feature Contribution (SHAP)")
            fig = plt.figure(figsize=(6, 4))
            features = preprocessor.get_feature_names_out() if hasattr(preprocessor, "get_feature_names_out") else None
            shap.plots.bar(shap.Explanation(values=shap_values[0], data=X_proc[0], feature_names=features), max_display=10, show=False)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"SHAP explanation failed: {e}")

        # Bug 4 fix: persist decision
        record = {
            "name": name,
            "age": int(age),
            "gender": gender,
            "monthly_income": income,
            "utility_bill": utility_bill,
            "repayment_history_pct": repayment_history,
            "employment_length": employment_length,
            "prob_default": round(prob_default, 4),
            "threshold": threshold,
            "decision": decision,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        if "audit_log" not in st.session_state:
            st.session_state.audit_log = []
        st.session_state.audit_log.append(record)

        try:
            from data_ingestion.mongo_loader import save_decision
            save_decision(record)
            st.caption("✅ Decision saved to MongoDB.")
        except Exception:
            st.caption("ℹ️ Decision saved to session log (MongoDB not configured).")

# ── Page: Dashboard ───────────────────────────────────────────────────────────

def page_dashboard():
    st.subheader("📊 Model Performance Overview")
    artifacts = _load_artifacts()
    if artifacts is None:
        st.info("No trained model found. Open **Threshold Optimizer** in the sidebar.")
        return

    _, _, y_test, y_prob, _ = artifacts
    auc = roc_auc_score(y_test, y_prob)

    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC", f"{auc:.4f}")
    c2.metric("Test Samples", f"{len(y_test):,}")
    c3.metric("Default Rate (test)", f"{y_test.mean():.1%}")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_frame = pd.DataFrame({
        "false_positive_rate": np.concatenate([fpr, np.array([0.0, 1.0])]),
        "true_positive_rate": np.concatenate([tpr, np.array([0.0, 1.0])]),
        "series": ["Model"] * len(fpr) + ["Baseline", "Baseline"],
    })
    chart = (
        alt.Chart(roc_frame)
        .mark_line(strokeWidth=3)
        .encode(
            x=alt.X("false_positive_rate", title="False Positive Rate"),
            y=alt.Y("true_positive_rate", title="True Positive Rate"),
            color=alt.Color("series", title="Curve"),
            strokeDash=alt.condition(alt.datum.series == "Baseline", alt.value([6, 4]), alt.value([1, 0])),
        )
        .properties(title=f"ROC Curve (AUC = {auc:.4f})", height=320)
    )
    st.altair_chart(chart, use_container_width=True)

# ── Page: Threshold Optimizer ─────────────────────────────────────────────────

def page_threshold_optimizer():
    st.subheader("🎯 Threshold Optimizer — Lender Rules Engine")
    st.markdown("---")
    st.markdown("#### Step 1 — Model Status")

    artifacts = _load_artifacts()

    if artifacts is None:
        if not os.path.exists(DATA_PATH):
            st.error("❌ Dataset not found.")
            return

        if st.button("🚀 Train Model"):
            try:
                with st.spinner("Training..."):
                    auc, threshold_info = _do_train()
                st.session_state["thresh_slider"] = float(threshold_info["threshold"])
                st.rerun()
            except Exception as exc:
                st.error(f"Training failed: {exc}")
        return

    _, _, y_test, y_prob, threshold_info = artifacts
    auc = roc_auc_score(y_test, y_prob)

    col_a, col_b, col_retrain = st.columns([2, 1, 1])
    col_a.success(f"✅ Model loaded — ROC-AUC = **{auc:.4f}**")
    if col_retrain.button("🔄 Retrain"):
        auc, threshold_info = _do_train()
        st.session_state["thresh_slider"] = float(threshold_info["threshold"])
        st.rerun()

    from evaluation.thresholds import (
        find_optimal_threshold,
        get_metrics_at_threshold,
        optimize_threshold_from_pr_curve,
    )

    sweep_df = _sweep_thresholds(tuple(y_test.tolist()), tuple(y_prob.tolist()))

    st.markdown("---")
    st.markdown("#### Step 2 — Set Decision Threshold")
    st.info(f"Recommended business threshold: **{threshold_info['threshold']:.2f}**")

    threshold = st.slider(
        "Decision Threshold",
        min_value=0.01, max_value=0.99, step=0.01,
        value=float(st.session_state.get("thresh_slider", float(threshold_info.get("threshold", 0.50)))),
        key="thresh_slider",
    )

    m = get_metrics_at_threshold(y_test, y_prob, threshold)
    st.markdown(f"#### Step 3 — Live Metrics at **{threshold:.2f}**")
    cols = st.columns(5)
    cols[0].metric("Precision", f"{m['precision']:.3f}")
    cols[1].metric("Recall", f"{m['recall']:.3f}")
    cols[2].metric("F1 Score", f"{m['f1']:.3f}")
    cols[3].metric("Accuracy", f"{m['accuracy']:.3f}")
    cols[4].metric("Approval Rate", f"{m['approval_rate']:.1%}")

    # Interactive Confusion Matrix
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    cm_frame = pd.DataFrame([
        {"actual": "Paid", "predicted": "Approved", "count": int(cm[0, 0])},
        {"actual": "Paid", "predicted": "Denied", "count": int(cm[0, 1])},
        {"actual": "Default", "predicted": "Approved", "count": int(cm[1, 0])},
        {"actual": "Default", "predicted": "Denied", "count": int(cm[1, 1])},
    ])
    heatmap = alt.Chart(cm_frame).mark_rect().encode(
        x="predicted:N", y="actual:N", color=alt.Color("count:Q", scale=alt.Scale(scheme="greens"))
    ).properties(height=220)
    labels = alt.Chart(cm_frame).mark_text(fontSize=14, fontWeight="bold").encode(
        x="predicted:N", y="actual:N", text="count:Q",
        color=alt.condition(alt.datum.count > cm.max() * 0.55, alt.value("white"), alt.value("black"))
    )
    st.altair_chart(heatmap + labels, use_container_width=True)

    # Business Optimizer
    st.markdown("#### Precision-Recall Business Optimizer")
    pr_c1, pr_c2, pr_c3 = st.columns(3)
    min_p = pr_c1.slider("Min Precision", 0.0, 1.0, 0.60)
    min_r = pr_c2.slider("Min Recall", 0.0, 1.0, 0.20)
    min_a = pr_c3.slider("Min Approval Rate", 0.0, 1.0, 0.20)

    pr_best = optimize_threshold_from_pr_curve(y_test, y_prob, min_precision=min_p, min_recall=min_r, min_approval_rate=min_a)
    if st.button("Apply PR-Optimized Threshold"):
        st.session_state["thresh_slider"] = float(pr_best["threshold"])
        st.rerun()

    # Optimal Table
    OBJECTIVES = {"Maximize F1": "f1", "Maximize Precision": "precision", "Maximize Recall": "recall", "Balanced": "balanced", "Maximize Profit": "profit"}
    rows = []
    opt_results = {}
    for label, obj in OBJECTIVES.items():
        best = find_optimal_threshold(y_test, y_prob, objective=obj)
        opt_results[obj] = best
        rows.append({"Objective": label, "Threshold": f"{best['threshold']:.2f}", "F1": f"{best['f1']:.3f}", "Approval": f"{best['approval_rate']:.1%}"})
    
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    btn_cols = st.columns(len(OBJECTIVES))
    for idx, (label, obj) in enumerate(OBJECTIVES.items()):
        if btn_cols[idx].button(label.split()[-1], key=f"btn_{obj}"):
            st.session_state["thresh_slider"] = float(opt_results[obj]["threshold"])
            st.rerun()

# ── Audit Logs ──────────────────────────────────────────────────────────

def page_audit_logs():
    st.subheader("📋 Audit Logs")
    logs = list(st.session_state.get("audit_log", []))
    try:
        from data_ingestion.mongo_loader import load_decisions
        mongo_records = load_decisions()
        if mongo_records: logs = mongo_records
    except Exception: pass

    if not logs:
        st.info("No decisions recorded yet.")
        return

    df = pd.DataFrame(logs)
    st.dataframe(df, use_container_width=True)
    st.download_button("⬇️ Download CSV", df.to_csv(index=False), "audit_log.csv", "text/csv")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="EquiLend AI", layout="wide", page_icon="⚖️")
    st.title("⚖️ EquiLend AI: Transparent Credit Scoring")

    choice = st.sidebar.selectbox("Navigation", ["New Application", "Dashboard", "Threshold Optimizer", "Audit Logs"])
    
    artifacts = _load_artifacts()
    rec_thresh = float(artifacts[4].get("threshold", 0.50)) if artifacts else 0.50
    active_thresh = st.session_state.get("thresh_slider", rec_thresh)
    st.sidebar.metric("Active Threshold", f"{active_thresh:.2f}")

    if choice == "New Application": page_new_application()
    elif choice == "Dashboard": page_dashboard()
    elif choice == "Threshold Optimizer": page_threshold_optimizer()
    elif choice == "Audit Logs": page_audit_logs()

if __name__ == "__main__":
    main()
```