"""
Customer Churn Intelligence Streamlit App.

Run from this folder with:
    streamlit run app.py

The training notebook should be run first so the models/ folder exists.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
CHURN_BASE_RATE = 0.265
DEFAULT_DECISION_THRESHOLD = 0.50

CONTRACT_OPTIONS = ["Month-to-month", "One year", "Two year"]
PAYMENT_OPTIONS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
INTERNET_OPTIONS = ["DSL", "Fiber optic", "No"]
YES_NO = ["Yes", "No"]

CONTRACT_MAP = {"Month-to-month": 0, "One year": 1, "Two year": 2}
PAYMENT_MAP = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3,
}
INTERNET_MAP = {"No": 0, "DSL": 1, "Fiber optic": 2}
SERVICE_MAP = {"No internet service": 0, "No": 1, "Yes": 2}
PHONE_LINE_MAP = {"No phone service": 0, "No": 1, "Yes": 2}

RAW_TEMPLATE_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    [data-testid="stMetricValue"] { font-size: 1.45rem !important; font-weight: 700; }
    [data-testid="stMetricLabel"] { color: #4b5563; }
    .status-box {
        border: 1px solid #d1d5db;
        border-left-width: 6px;
        border-radius: 8px;
        padding: 14px 16px;
        margin: 8px 0 14px 0;
        background: #ffffff;
    }
    .status-high { border-left-color: #dc2626; }
    .status-medium { border-left-color: #d97706; }
    .status-low { border-left-color: #16a34a; }
    .muted { color: #6b7280; font-size: 0.92rem; }
    .small-note { color: #6b7280; font-size: 0.84rem; }
    footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts():
    if not MODEL_DIR.exists():
        st.error(
            f"Missing models folder: {MODEL_DIR}. Run the training notebook first."
        )
        st.stop()

    model_path = next(
        (
            path
            for path in [
                MODEL_DIR / "deployed_model.pkl",
                MODEL_DIR / "best_model.pkl",
                MODEL_DIR / "xgb_model.pkl",
            ]
            if path.exists()
        ),
        None,
    )
    required = [model_path, MODEL_DIR / "scaler.pkl", MODEL_DIR / "feature_names.json"]
    missing = [str(path) for path in required if path is None or not path.exists()]
    if missing:
        st.error("Missing required model artifacts:\n\n" + "\n".join(missing))
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    feature_names = load_json(MODEL_DIR / "feature_names.json", [])
    metadata = load_json(MODEL_DIR / "preprocessing_metadata.json", {})
    comparison = None
    comparison_path = MODEL_DIR / "model_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)

    metadata.setdefault("model_path", model_path.name)
    metadata.setdefault("deployed_model_name", model_path.stem)
    metadata.setdefault("decision_threshold", DEFAULT_DECISION_THRESHOLD)
    metadata.setdefault("monthly_charge_mean", 64.76)
    metadata.setdefault("high_charge_threshold", 72.0)
    metadata.setdefault("metadata_available", (MODEL_DIR / "preprocessing_metadata.json").exists())
    return model, scaler, feature_names, metadata, comparison


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_int(value: Any, default: int = 0) -> int:
    return int(round(coerce_float(value, float(default))))


def yes_no_value(value: Any, default: str = "No") -> str:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"yes", "y", "true", "1"}:
            return "Yes"
        if text in {"no", "n", "false", "0"}:
            return "No"
    if value in {1, True}:
        return "Yes"
    if value in {0, False}:
        return "No"
    return default


def option_value(value: Any, options: list[str], default: str) -> str:
    if isinstance(value, str):
        text = " ".join(value.strip().split()).lower()
        for option in options:
            if text == option.lower():
                return option
    return default


def normalize_customer(raw: dict[str, Any]) -> dict[str, Any]:
    tenure = min(max(coerce_int(raw.get("tenure"), 0), 0), 72)
    monthly = min(max(coerce_float(raw.get("MonthlyCharges", raw.get("monthly_charges")), 65.0), 0.0), 500.0)
    total = coerce_float(raw.get("TotalCharges", raw.get("total_charges")), tenure * monthly)
    if total <= 0 and tenure > 0:
        total = tenure * monthly

    gender = option_value(raw.get("gender"), ["Male", "Female"], "Male")
    senior = yes_no_value(raw.get("SeniorCitizen", raw.get("senior_citizen")), "No")
    partner = yes_no_value(raw.get("Partner", raw.get("partner")), "No")
    dependents = yes_no_value(raw.get("Dependents", raw.get("dependents")), "No")
    phone = yes_no_value(raw.get("PhoneService", raw.get("phone_service")), "Yes")
    internet = option_value(raw.get("InternetService", raw.get("internet_svc")), INTERNET_OPTIONS, "Fiber optic")

    multiple_lines = option_value(
        raw.get("MultipleLines", raw.get("multiple_lines")),
        ["No", "Yes", "No phone service"],
        "No",
    )
    if phone == "No":
        multiple_lines = "No phone service"

    def service(name: str, fallback_key: str) -> str:
        value = option_value(
            raw.get(name, raw.get(fallback_key)),
            ["No", "Yes", "No internet service"],
            "No",
        )
        return "No internet service" if internet == "No" else value

    return {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": max(total, 0.0),
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": service("OnlineSecurity", "online_security"),
        "OnlineBackup": service("OnlineBackup", "online_backup"),
        "DeviceProtection": service("DeviceProtection", "device_prot"),
        "TechSupport": service("TechSupport", "tech_support"),
        "StreamingTV": service("StreamingTV", "streaming_tv"),
        "StreamingMovies": service("StreamingMovies", "streaming_movies"),
        "Contract": option_value(raw.get("Contract", raw.get("contract")), CONTRACT_OPTIONS, "Month-to-month"),
        "PaperlessBilling": yes_no_value(raw.get("PaperlessBilling", raw.get("paperless")), "Yes"),
        "PaymentMethod": option_value(raw.get("PaymentMethod", raw.get("payment_method")), PAYMENT_OPTIONS, "Electronic check"),
    }


def cohort_average_for_tenure(metadata: dict[str, Any], tenure: int) -> float:
    cohort = metadata.get("cohort_monthly_charge_by_tenure") or metadata.get("cohort_avg_by_tenure") or {}
    if not cohort:
        return float(metadata.get("monthly_charge_mean", 64.76))

    string_key = str(int(tenure))
    if string_key in cohort:
        return coerce_float(cohort[string_key], metadata.get("monthly_charge_mean", 64.76))

    numeric_keys = sorted(coerce_int(key) for key in cohort.keys())
    if not numeric_keys:
        return float(metadata.get("monthly_charge_mean", 64.76))
    nearest = min(numeric_keys, key=lambda key: abs(key - tenure))
    return coerce_float(cohort.get(str(nearest)), metadata.get("monthly_charge_mean", 64.76))


def build_feature_frame(
    customer: dict[str, Any],
    feature_names: list[str],
    metadata: dict[str, Any],
) -> pd.DataFrame:
    customer = normalize_customer(customer)
    services = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    tenure = customer["tenure"]
    monthly = customer["MonthlyCharges"]
    total = customer["TotalCharges"]
    high_charge_threshold = coerce_float(metadata.get("high_charge_threshold"), 72.0)
    spend_baseline = cohort_average_for_tenure(metadata, tenure)

    row = {
        "gender": int(customer["gender"] == "Male"),
        "SeniorCitizen": int(customer["SeniorCitizen"] == "Yes"),
        "Partner": int(customer["Partner"] == "Yes"),
        "Dependents": int(customer["Dependents"] == "Yes"),
        "tenure": tenure,
        "PhoneService": int(customer["PhoneService"] == "Yes"),
        "MultipleLines": PHONE_LINE_MAP[customer["MultipleLines"]],
        "InternetService": INTERNET_MAP[customer["InternetService"]],
        "OnlineSecurity": SERVICE_MAP[customer["OnlineSecurity"]],
        "OnlineBackup": SERVICE_MAP[customer["OnlineBackup"]],
        "DeviceProtection": SERVICE_MAP[customer["DeviceProtection"]],
        "TechSupport": SERVICE_MAP[customer["TechSupport"]],
        "StreamingTV": SERVICE_MAP[customer["StreamingTV"]],
        "StreamingMovies": SERVICE_MAP[customer["StreamingMovies"]],
        "Contract": CONTRACT_MAP[customer["Contract"]],
        "PaperlessBilling": int(customer["PaperlessBilling"] == "Yes"),
        "PaymentMethod": PAYMENT_MAP[customer["PaymentMethod"]],
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "AvgMonthlySpend": total / (tenure + 1),
        "SpendPremium": monthly - spend_baseline,
        "NumAddOnServices": sum(customer[col] == "Yes" for col in services),
        "HasBundle": int(customer["PhoneService"] == "Yes" and customer["InternetService"] in {"DSL", "Fiber optic"}),
        "HasFamily": int(customer["Partner"] == "Yes" or customer["Dependents"] == "Yes"),
        "ContractRisk": CONTRACT_MAP[customer["Contract"]],
        "AutoPay": int("automatic" in customer["PaymentMethod"].lower()),
        "HighChargesNewCustomer": int(monthly > high_charge_threshold and tenure < 12),
        "Tenure_sq": tenure**2,
        "LogTotalCharges": np.log1p(total),
        "LogMonthlyCharges": np.log1p(monthly),
    }
    frame = pd.DataFrame([row])
    for feature in feature_names:
        if feature not in frame.columns:
            frame[feature] = 0
    return frame[feature_names].astype(float)


def predict_probability(model: Any, matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(matrix)[:, 1]
    prediction = model.predict(matrix)
    return np.asarray(prediction).reshape(-1)


def risk_level(probability: float) -> str:
    if probability >= 0.60:
        return "High"
    if probability >= 0.35:
        return "Medium"
    return "Low"


def gauge_chart(probability: float, threshold: float) -> go.Figure:
    value = probability * 100
    threshold_value = threshold * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=value,
            number={"suffix": "%", "font": {"size": 40}},
            delta={
                "reference": CHURN_BASE_RATE * 100,
                "suffix": "%",
                "relative": False,
                "increasing": {"color": "#dc2626"},
                "decreasing": {"color": "#16a34a"},
            },
            title={"text": "Predicted churn probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#dc2626" if probability >= threshold else "#16a34a"},
                "steps": [
                    {"range": [0, 35], "color": "#dcfce7"},
                    {"range": [35, 60], "color": "#fef3c7"},
                    {"range": [60, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#111827", "width": 3},
                    "thickness": 0.7,
                    "value": threshold_value,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=35, b=10))
    return fig


def risk_signals(customer: dict[str, Any]) -> tuple[list[str], list[str]]:
    risks: list[str] = []
    strengths: list[str] = []
    if customer["Contract"] == "Month-to-month":
        risks.append("Month-to-month contract keeps switching cost low.")
    elif customer["Contract"] == "Two year":
        strengths.append("Two-year contract is a strong retention signal.")

    if customer["tenure"] < 12:
        risks.append("Customer is in the first year, where churn is usually highest.")
    elif customer["tenure"] >= 36:
        strengths.append("Long tenure suggests an established relationship.")

    if customer["MonthlyCharges"] > 80:
        risks.append("Monthly charge is high compared with the portfolio.")
    if customer["InternetService"] == "Fiber optic":
        risks.append("Fiber optic customers are a known high-churn segment in this dataset.")
    if customer["OnlineSecurity"] == "No":
        risks.append("No Online Security add-on.")
    if customer["TechSupport"] == "No":
        risks.append("No Tech Support add-on.")
    if customer["PaymentMethod"] == "Electronic check":
        risks.append("Electronic check is associated with higher churn.")
    if "automatic" in customer["PaymentMethod"].lower():
        strengths.append("Automatic payment reduces payment friction.")
    if customer["Partner"] == "Yes" or customer["Dependents"] == "Yes":
        strengths.append("Partner/dependents indicate household attachment.")
    return risks, strengths


def recommended_actions(customer: dict[str, Any], probability: float) -> list[tuple[str, str]]:
    actions: list[tuple[str, str]] = []
    if customer["Contract"] == "Month-to-month":
        actions.append(("Contract upgrade", "Offer a low-friction 12-month contract with a limited discount."))
    if customer["MonthlyCharges"] > 70 and customer["tenure"] < 18:
        actions.append(("Price reassurance", "Offer a price-lock or bill review for high-value newer customers."))
    if customer["OnlineSecurity"] == "No" or customer["TechSupport"] == "No":
        actions.append(("Support bundle", "Offer Online Security and Tech Support as a small bundle."))
    if customer["PaymentMethod"] == "Electronic check":
        actions.append(("Auto-pay migration", "Offer a small incentive to move to automatic payment."))
    if probability < 0.35:
        actions.append(("Loyalty nudge", "Keep the customer warm with a light-touch loyalty message."))
    return actions or [("Review account", "No obvious rule-based action; use call notes and customer history.")]


def render_sidebar() -> dict[str, Any]:
    st.sidebar.title("Customer profile")
    with st.sidebar.expander("Account", expanded=True):
        tenure = st.slider("Tenure in months", 0, 72, 12)
        contract = st.selectbox("Contract", CONTRACT_OPTIONS)
        payment_method = st.selectbox("Payment method", PAYMENT_OPTIONS)
        paperless = st.radio("Paperless billing", YES_NO, horizontal=True)

    with st.sidebar.expander("Charges", expanded=True):
        monthly = st.slider("Monthly charges", 18.0, 120.0, 65.0, 0.50)
        default_total = float(max(0.0, tenure * monthly))
        total = st.number_input("Total charges", 0.0, 10000.0, default_total, 10.0)

    with st.sidebar.expander("Demographics", expanded=False):
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        senior = st.radio("Senior citizen", YES_NO, horizontal=True)
        partner = st.radio("Partner", YES_NO, horizontal=True)
        dependents = st.radio("Dependents", YES_NO, horizontal=True)

    with st.sidebar.expander("Services", expanded=False):
        phone = st.radio("Phone service", YES_NO, horizontal=True)
        line_options = ["No phone service"] if phone == "No" else ["No", "Yes"]
        multiple_lines = st.selectbox("Multiple lines", line_options)
        internet = st.selectbox("Internet service", INTERNET_OPTIONS)
        service_options = ["No internet service"] if internet == "No" else ["No", "Yes"]
        online_security = st.selectbox("Online security", service_options)
        online_backup = st.selectbox("Online backup", service_options)
        device_protection = st.selectbox("Device protection", service_options)
        tech_support = st.selectbox("Tech support", service_options)
        streaming_tv = st.selectbox("Streaming TV", service_options)
        streaming_movies = st.selectbox("Streaming movies", service_options)

    return normalize_customer(
        {
            "tenure": tenure,
            "Contract": contract,
            "PaymentMethod": payment_method,
            "PaperlessBilling": paperless,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
        }
    )


def score_customer(model: Any, scaler: Any, feature_names: list[str], metadata: dict[str, Any], customer: dict[str, Any]):
    feature_frame = build_feature_frame(customer, feature_names, metadata)
    scaled = scaler.transform(feature_frame)
    probability = float(predict_probability(model, scaled)[0])
    threshold = coerce_float(metadata.get("decision_threshold"), DEFAULT_DECISION_THRESHOLD)
    prediction = int(probability >= threshold)
    return probability, prediction, feature_frame


def render_prediction_tab(model: Any, scaler: Any, feature_names: list[str], metadata: dict[str, Any], customer: dict[str, Any]):
    probability, prediction, feature_frame = score_customer(model, scaler, feature_names, metadata, customer)
    level = risk_level(probability)
    threshold = coerce_float(metadata.get("decision_threshold"), DEFAULT_DECISION_THRESHOLD)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Churn probability",
        f"{probability * 100:.1f}%",
        f"{(probability - CHURN_BASE_RATE) * 100:+.1f}% vs base",
        delta_color="inverse",
    )
    c2.metric("Risk level", level)
    c3.metric("Decision", "Contact" if prediction else "Monitor")
    c4.metric("Tenure", f"{customer['tenure']} mo")
    c5.metric("Monthly charge", f"${customer['MonthlyCharges']:.0f}")

    st.markdown("---")
    left, middle, right = st.columns([1.15, 1.2, 1.2])

    with left:
        st.plotly_chart(gauge_chart(probability, threshold), use_container_width=True)
        status_class = f"status-{level.lower()}"
        st.markdown(
            f"""
            <div class="status-box {status_class}">
                <b>{level} churn risk</b><br>
                <span class="muted">Decision threshold: {threshold:.2f}. Base churn rate: {CHURN_BASE_RATE:.1%}.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with middle:
        st.subheader("Risk signals")
        risks, strengths = risk_signals(customer)
        if risks:
            for item in risks:
                st.warning(item)
        else:
            st.success("No major rule-based risk signal found.")
        if strengths:
            st.subheader("Protective signals")
            for item in strengths:
                st.info(item)

    with right:
        st.subheader("Recommended actions")
        for title, text in recommended_actions(customer, probability):
            st.markdown(f"**{title}**")
            st.write(text)
        expected_months = max(3, 12 - customer["tenure"]) if prediction else 24
        estimated_ltv = customer["MonthlyCharges"] * expected_months
        st.metric("Simple LTV estimate", f"${estimated_ltv:,.0f}", f"{expected_months} projected months")

    with st.expander("Encoded model input"):
        st.dataframe(feature_frame.T.rename(columns={0: "Value"}), use_container_width=True)


def render_performance_tab(metadata: dict[str, Any], comparison: pd.DataFrame | None):
    st.subheader("Model performance")
    if comparison is None or comparison.empty:
        st.info("Run the notebook to generate models/model_comparison.csv.")
        return

    st.dataframe(comparison, use_container_width=True)
    if "ROC-AUC" in comparison.columns:
        best_row = comparison.sort_values("ROC-AUC", ascending=False).iloc[0]
        st.success(f"Best saved comparison by ROC-AUC: {best_row['Model']} ({best_row['ROC-AUC']:.4f}).")

    deployed = metadata.get("deployed_model_name", metadata.get("model_path", "unknown"))
    st.info(f"Model currently loaded by the app: {deployed}.")
    if not metadata.get("metadata_available", False):
        st.warning(
            "preprocessing_metadata.json is missing. The app is using fallback training constants. "
            "Rerun the updated notebook to save exact deployment metadata."
        )

    metrics = [col for col in ["Accuracy", "Recall", "Precision", "F1-Score", "ROC-AUC", "Brier Score"] if col in comparison.columns]
    if metrics and "Model" in comparison.columns:
        chart_df = comparison.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score")
        fig = px.bar(chart_df, x="Score", y="Model", color="Metric", barmode="group", orientation="h")
        fig.update_layout(height=max(360, 44 * len(comparison)), yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("How to read the metrics")
    c1, c2, c3 = st.columns(3)
    c1.write("Recall matters because missing a true churner can be expensive.")
    c2.write("ROC-AUC measures ranking quality across thresholds.")
    c3.write("Brier score checks whether predicted probabilities are calibrated.")


def render_batch_tab(model: Any, scaler: Any, feature_names: list[str], metadata: dict[str, Any]):
    st.subheader("Batch scoring")
    st.write("Upload customers with the same raw columns as the IBM Telco dataset.")
    template = pd.DataFrame(
        [
            {
                "tenure": 12,
                "MonthlyCharges": 65.0,
                "TotalCharges": 780.0,
                "gender": "Male",
                "SeniorCitizen": "No",
                "Partner": "No",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
            }
        ]
    )
    st.download_button(
        "Download CSV template",
        template.to_csv(index=False),
        file_name="churn_batch_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded is None:
        return

    batch = pd.read_csv(uploaded)
    missing = [col for col in RAW_TEMPLATE_COLUMNS if col not in batch.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        return

    customers = [normalize_customer(row.to_dict()) for _, row in batch.iterrows()]
    frames = [build_feature_frame(customer, feature_names, metadata) for customer in customers]
    feature_matrix = pd.concat(frames, ignore_index=True)
    probabilities = predict_probability(model, scaler.transform(feature_matrix))
    threshold = coerce_float(metadata.get("decision_threshold"), DEFAULT_DECISION_THRESHOLD)

    result = batch.copy()
    result["ChurnProbability"] = np.round(probabilities * 100, 1)
    result["RiskLevel"] = [risk_level(float(prob)) for prob in probabilities]
    result["RetentionDecision"] = np.where(probabilities >= threshold, "Contact", "Monitor")
    result = result.sort_values("ChurnProbability", ascending=False)

    st.success(f"Scored {len(result):,} customers.")
    c1, c2, c3 = st.columns(3)
    c1.metric("High risk", int((result["RiskLevel"] == "High").sum()))
    c2.metric("Medium risk", int((result["RiskLevel"] == "Medium").sum()))
    c3.metric("Low risk", int((result["RiskLevel"] == "Low").sum()))

    fig = px.histogram(result, x="ChurnProbability", nbins=30, title="Churn probability distribution")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(result.head(100), use_container_width=True)
    st.download_button(
        "Download scored results",
        result.to_csv(index=False),
        file_name="churn_predictions.csv",
        mime="text/csv",
    )


def render_about_tab():
    st.subheader("Project methodology")
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
**Dataset**

- IBM Telco Customer Churn
- 7,043 customers
- Binary target: churned or retained
- Public, cross-sectional dataset

**Core workflow**

- Clean `TotalCharges`
- Drop `customerID`
- Engineer service, contract, payment, tenure, and spending features
- Train several classifiers
- Compare using ROC-AUC, recall, F1, precision, and Brier score
- Save artifacts for Streamlit deployment
"""
        )
    with right:
        st.markdown(
            """
**Deployment notes**

- The app loads artifacts from the local `models/` folder.
- Single-customer and batch scoring now share the same preprocessing path.
- If `preprocessing_metadata.json` exists, exact training constants are used.
- If metadata is missing, safe fallback constants are used and a warning is shown.

**Limitations**

- This is not causal analysis.
- Business actions need real campaign response data.
- Public dataset probabilities may not match a real telecom portfolio.
- Final threshold should be chosen on validation data with real retention costs.
"""
        )


def main():
    model, scaler, feature_names, metadata, comparison = load_artifacts()
    customer = render_sidebar()

    st.title("Customer Churn Intelligence")
    st.caption("Predict churn risk, explain the main drivers, and prioritize retention actions.")

    tabs = st.tabs(["Prediction", "Model performance", "Batch scoring", "About"])
    with tabs[0]:
        render_prediction_tab(model, scaler, feature_names, metadata, customer)
    with tabs[1]:
        render_performance_tab(metadata, comparison)
    with tabs[2]:
        render_batch_tab(model, scaler, feature_names, metadata)
    with tabs[3]:
        render_about_tab()


if __name__ == "__main__":
    main()
