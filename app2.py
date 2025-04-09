import streamlit as st
import pandas as pd
import plotly.express as px
import re

# Load Data
df = pd.read_excel("BaselineCombined.xlsx")
df["Participant"] = df["Name"] + " (" + df["District"] + ")"

# Categorize Question Type
df["Question Type"] = df["Remark"].apply(
    lambda x: "Knowledge-based" if str(x).strip().lower() in ["correct", "incorrect"] else "Open-ended"
)

# Detect Multiple Selections in Responses
def detect_multiple(res):
    if pd.isna(res) or not isinstance(res, str):
        return "Unclear"
    # If separated by commas, slashes, or line breaks
    if re.search(r"[,/\n]", res):
        return "Yes"
    return "No"

df["Multiple Responses?"] = df["Responses"].apply(detect_multiple)

st.set_page_config(page_title="Baseline Dashboard", layout="wide")
st.title("📊 Baseline Survey Analysis Dashboard")

# Sidebar filters
with st.sidebar:
    st.header("🔍 Filters")
    districts = st.multiselect("Select District(s)", sorted(df["District"].unique()), default=None)
    questions = st.multiselect("Select Question(s)", sorted(df["Question"].unique()), default=None)

# Filtered Data
filtered_df = df.copy()
if districts:
    filtered_df = filtered_df[filtered_df["District"].isin(districts)]
if questions:
    filtered_df = filtered_df[filtered_df["Question"].isin(questions)]

# Section: Knowledge-based Analysis
st.subheader("✅ Knowledge-based Question Summary")
kb_df = filtered_df[filtered_df["Question Type"] == "Knowledge-based"]

if not kb_df.empty:
    summary = kb_df.groupby(["District", "Remark"]).size().reset_index(name="Count")
    fig = px.bar(summary, x="District", y="Count", color="Remark", barmode="group",
                title="Correct vs Incorrect by District")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No knowledge-based questions in the current filter.")

# Section: Option Selection Analysis
st.subheader("📌 Most Chosen Options (for Open-ended / Non-evaluated Questions)")
op_df = filtered_df[filtered_df["Question Type"] == "Open-ended"]

if not op_df.empty:
    for q in op_df["Question"].unique():
        st.markdown(f"**Question:** {q}")
        q_df = op_df[op_df["Question"] == q]
        count_df = q_df["Responses"].value_counts(normalize=True).reset_index()
        count_df.columns = ["Response", "Percentage"]
        count_df["Percentage"] *= 100
        fig = px.bar(count_df, x="Response", y="Percentage", title="Response Distribution",
                    labels={"Percentage": "% of Participants"})
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No open-ended questions in the current filter.")

# Section: Participant-level View
st.subheader("🧑 Participant Response Table")
st.dataframe(filtered_df[["Participant", "Question", "Responses", "Remark", "Multiple Responses?"]], use_container_width=True)

# Optional: Export filtered data
with st.expander("⬇ Export Filtered Data"):
    st.download_button("Download CSV", filtered_df.to_csv(index=False), "filtered_data.csv")
