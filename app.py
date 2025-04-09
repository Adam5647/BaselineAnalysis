import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# Set Streamlit page layout
st.set_page_config(layout="wide")
st.title("📊 Baseline Survey Analysis Dashboard")

# Load data
@st.cache_data

def load_data():
    df = pd.read_excel("BaselineCombined.xlsx", sheet_name="Combined")
    df['Remark'] = df['Remark'].astype(str).str.lower().str.strip()
    df['Question'] = df['Question'].astype(str).str.strip()
    df['Responses'] = df['Responses'].astype(str).str.strip()
    df['Participant'] = df['District'].astype(str) + " - " + df['Name'].astype(str)
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
districts = st.sidebar.multiselect(
    "Select District(s):",
    options=df['District'].unique(),
    default=df['District'].unique()
)
filtered_df = df[df['District'].isin(districts)]

# Separate data types
knowledge_df = filtered_df[filtered_df['Remark'].isin(['correct', 'incorrect'])]
subjective_df = filtered_df[~filtered_df['Remark'].isin(['correct', 'incorrect'])]

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Knowledge-Based Analysis",
    "Open-Ended / Subjective",
    "Most Selected Options",
    "🧠 AI Insights with LLaMA Mistral"
])

# Tab 1: Knowledge-Based
with tab1:
    st.subheader("✅ Knowledge-Based Questions (Correct vs Incorrect)")

    # District-level summary
    district_summary = knowledge_df.groupby(['District', 'Remark']).size().unstack(fill_value=0)
    district_summary['Total'] = district_summary.sum(axis=1)
    district_summary['% Correct'] = (district_summary.get('correct', 0) / district_summary['Total']) * 100
    district_summary['% Incorrect'] = (district_summary.get('incorrect', 0) / district_summary['Total']) * 100

    st.write("### District-wise Summary")
    st.dataframe(district_summary.style.format("{:.2f}"))

    st.write("### Correct vs Incorrect (District-wise)")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    district_summary[['correct', 'incorrect']].plot(kind='bar', stacked=True, ax=ax1, color=["#4CAF50", "#F44336"])
    ax1.set_ylabel("Number of Responses")
    ax1.set_title("Correct vs Incorrect Answers by District")
    st.pyplot(fig1)

    # Question-level summary
    st.write("### Question-wise Summary")
    q_summary = knowledge_df.groupby(['Question', 'Remark']).size().unstack(fill_value=0)
    q_summary['Total'] = q_summary.sum(axis=1)
    q_summary['% Correct'] = (q_summary.get('correct', 0) / q_summary['Total']) * 100
    st.dataframe(q_summary.style.format("{:.2f}"))

    # Specific analysis for Question 3 and 6
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Question 3: Standing Committees")
        q3_text = "3. What are the standing committees that must be in place in Gram Panchayats? [Mark all the correct answers]"
        correct_q3 = {
            "3. General Standing Committee",
            "3. Finance, Audit and Planning Committee",
            "3. Social Justice Committee"
        }
        q3_df = df[df['Question'] == q3_text].copy()
        q3_df['Correct'] = q3_df['Responses'].apply(lambda x: x in correct_q3)
        grouped_q3 = q3_df.groupby('Participant')['Responses'].agg(list).reset_index()
        grouped_q3['Fully Correct (All 3 Selected)'] = grouped_q3['Responses'].apply(lambda x: set(x) == correct_q3)
        grouped_q3['% Correct Selections'] = grouped_q3['Responses'].apply(lambda x: 100 * (len(set(x) & correct_q3) / len(correct_q3)))
        st.dataframe(grouped_q3.set_index('Participant').style.format({"% Correct Selections": "{:.2f}"}))

    with col2:
        st.write("### Question 6: Sources of Finance")
        q6_text = "6. What are the sources of finance for the Gram Panchayat? [Select all correct answers]"
        correct_q6 = {
            "6. Central government grants under schemes like MGNREGS.",
            "6. Taxes collected by the Panchayat such as property tax and water tax.",
            "6. Donations from individuals or organizations.",
            "6. Income from Panchayat-owned assets such as markets or community halls.",
            "6. State government grants.",
            "6. Fines and penalties imposed by the Panchayat."
        }
        invalid_q6 = {
            "6. None of the above answers are correct.",
            "6. I have no information about this."
        }
        q6_df = df[df['Question'] == q6_text].copy()
        grouped_q6 = q6_df.groupby('Participant')['Responses'].agg(list).reset_index()
        grouped_q6['Invalid Answer'] = grouped_q6['Responses'].apply(lambda x: any(opt in x for opt in invalid_q6))
        grouped_q6['Fully Correct (All 6 Selected)'] = grouped_q6.apply(lambda row: set(row['Responses']) == correct_q6 and not row['Invalid Answer'], axis=1)
        grouped_q6['% Correct Selections'] = grouped_q6['Responses'].apply(lambda x: 100 * (len(set(x) & correct_q6) / len(correct_q6)))
        st.dataframe(grouped_q6.set_index('Participant').style.format({"% Correct Selections": "{:.2f}"}))

# Tab 2: Open-ended/Subjective
with tab2:
    st.subheader("🗣️ Open-Ended / Subjective Questions")
    question_choice = st.selectbox("Select Question:", subjective_df['Question'].unique())
    selected_q = subjective_df[subjective_df['Question'] == question_choice]

    response_counts = selected_q['Responses'].value_counts(normalize=True) * 100

    st.write("### Response Distribution (%)")
    st.dataframe(
        response_counts.rename("Percentage")
        .reset_index()
        .rename(columns={"index": "Response"})
        .style.format({"Percentage": "{:.2f}"})
    )

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.barplot(y=response_counts.index, x=response_counts.values, ax=ax2, palette="Blues_d")
    ax2.set_xlabel("Percentage")
    ax2.set_ylabel("Responses")
    ax2.set_title("Response Distribution")
    st.pyplot(fig2)

# Tab 3: Option Frequency for Opinion Questions
with tab3:
    st.subheader("📌 Most Chosen Options (General Opinions)")
    opinion_df = subjective_df.copy()

    question_options = opinion_df['Question'].unique()
    question_selected = st.selectbox("Select Question for Option Analysis:", question_options)
    selected_data = opinion_df[opinion_df['Question'] == question_selected]

    option_counts = selected_data['Responses'].value_counts(normalize=True) * 100

    st.write("### Option Frequency (%)")
    st.dataframe(
        option_counts.rename("Percentage")
        .reset_index()
        .rename(columns={"index": "Option"})
        .style.format({"Percentage": "{:.2f}"})
    )

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.barplot(y=option_counts.index, x=option_counts.values, ax=ax3, palette="Greens_d")
    ax3.set_xlabel("Percentage")
    ax3.set_ylabel("Options")
    ax3.set_title("Most Chosen Options")
    st.pyplot(fig3)

# Tab 4: AI Insights
with tab4:
    st.subheader("🧠 AI Insights with LLaMA Mistral")
    participant_selected = st.selectbox("Select Participant (District - Name):", df['Participant'].unique())
    person_df = df[df['Participant'] == participant_selected]

    if not person_df.empty:
        all_answers = "\n".join(
            f"Q: {row['Question']}\nA: {row['Responses']}" for _, row in person_df.iterrows()
        )

        prompt = (
            f"You are an expert data analyst. Analyze the following responses by a survey participant and provide a summary "
            f"that includes their general knowledge, opinions, and understanding level. Be concise and insightful.\n\n"
            f"{all_answers}"
        )

        st.write("### Raw Participant Responses")
        st.code(all_answers, language="text")

        if st.button("🔍 Generate Insight with LLaMA"):
            try:
                # Updated URL for the exposed Ollama model via Ngrok
                response = requests.post(
                    "https://44bd-2405-201-ac0b-e0cb-a4f0-149d-8b8-1220.ngrok-free.app/api/generate",
                    json={"model": "mistral", "prompt": prompt, "stream": False},
                    timeout=60
                )
                if response.status_code == 200:
                    result = response.json()
                    st.success("Insight generated successfully:")
                    st.markdown(result.get("response", "No response returned."))
                else:
                    st.error(f"LLaMA API returned an error: {response.status_code}\n{response.text}")
            except Exception as e:
                st.error(f"Failed to connect to LLaMA: {e}")
