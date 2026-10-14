import json
import os
import urllib.error
import urllib.request

import pandas as pd
import streamlit as st

st.set_page_config(page_title="DataTrust AI", page_icon="🛡️", layout="wide")

st.title("🛡️ DataTrust AI")
st.caption("Evidence-first data quality analysis • AWS Lambda + Amazon Bedrock")

with st.sidebar:
    st.subheader("Analysis")
    endpoint = st.text_input(
        "Lambda Function URL",
        value=os.environ.get("DATATRUST_API_URL", ""),
        help="Paste your deployed Lambda Function URL. Do not paste AWS access keys here.",
    )
    st.info("Your CSV is converted into a limited JSON sample and sent to your configured endpoint. Do not upload sensitive or regulated data to this demo.")

sample = pd.DataFrame({
    "customer_id": [101, 102, 103, 104, 104, 106],
    "age": [25, 38, None, 44, 44, 29],
    "monthly_spend": [1200, 2400, 1800, None, None, 900],
    "plan": ["basic", "premium", "basic", "premium", "premium", "basic"],
    "churned": [0, 1, 0, 1, 1, 0],
})

uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])
use_sample = st.button("Try the sample dataset", type="secondary")

df = None
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}")
elif use_sample:
    df = sample.copy()
    st.session_state["use_sample"] = True
elif st.session_state.get("use_sample"):
    df = sample.copy()

if df is not None:
    st.subheader("Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"{len(df):,} rows × {len(df.columns):,} columns")

    if len(df) > 5000:
        st.warning("For this demo, only the first 5,000 rows are supported. Upload a smaller sample.")
    elif len(df.columns) > 100:
        st.warning("For this demo, datasets are limited to 100 columns.")
    elif st.button("Analyze dataset", type="primary"):
        if not endpoint.strip().startswith("https://"):
            st.error("Add your HTTPS Lambda Function URL in the sidebar first.")
        elif df.empty or len(df.columns) == 0:
            st.error("The dataset is empty.")
        else:
            records = json.loads(df.head(5000).to_json(orient="records", date_format="iso"))
            body = json.dumps({"records": records}).encode("utf-8")
            request = urllib.request.Request(
                endpoint.strip(),
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with st.spinner("Profiling the dataset and preparing an evidence-based explanation…"):
                    with urllib.request.urlopen(request, timeout=90) as result:
                        data = json.loads(result.read().decode("utf-8"))
                if result.status != 200:
                    st.error(data.get("error", "Analysis failed."))
                else:
                    profile = data["profile"]
                    st.session_state["last_result"] = data
            except urllib.error.HTTPError as exc:
                try:
                    detail = exc.read().decode("utf-8")
                except Exception:
                    detail = str(exc)
                st.error(f"Backend returned HTTP {exc.code}: {detail[:1000]}")
            except Exception as exc:
                st.error(f"Could not reach the agent: {exc}")

result = st.session_state.get("last_result")
if result:
    profile = result["profile"]
    st.subheader("Data quality overview")
    a, b, c, d = st.columns(4)
    a.metric("Rows analyzed", profile["row_count"])
    b.metric("Columns", profile["column_count"])
    c.metric("Duplicate rows", profile["duplicate_row_count"])
    d.metric("Issues flagged", len(profile["issues"]))

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Column-level findings")
        report = pd.DataFrame(profile["columns"])
        report["numeric_summary"] = report["numeric_summary"].apply(
            lambda value: json.dumps(value) if value else "—"
        )
        st.dataframe(report, use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### Detected issues")
        if profile["issues"]:
            st.dataframe(pd.DataFrame(profile["issues"]), use_container_width=True, hide_index=True)
        else:
            st.success("No issues from the current rule set were detected. This does not prove the dataset is error-free.")

    st.markdown("#### DataTrust AI recommendations")
    st.markdown(result.get("explanation", "No explanation returned."))
    st.download_button(
        "Download profile JSON",
        data=json.dumps(profile, indent=2),
        file_name="datatrust_profile.json",
        mime="application/json",
    )
    st.caption("Metrics are computed by deterministic Python checks. The model provides recommendations and may still make mistakes; validate suggestions before changing data.")
