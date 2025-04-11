import streamlit as st
import pandas as pd
import io
import re
import requests

# --- LLM CALLERS ---
def call_openai(api_key, prompt):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a QA test expert."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def call_ollama(endpoint, model, prompt):
    response = requests.post(
        f"{endpoint}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
    )

    return response.json()["message"]["content"].strip()

def call_groq(endpoint, api_key, model, prompt):
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    json_data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(endpoint, headers=headers, json=json_data)
    return response.json()["choices"][0]["message"]["content"].strip()

# --- PARSER ---
def parse_llm_response(response):
    titles = re.findall(r"(?:Title|Test Case):\s*(.*)", response)
    steps_raw = re.findall(r"Steps:\n((?:- .+\n?)+)", response)
    expected = re.findall(r"Expected Result:\s*(.*)", response)

    rows = []
    for i in range(len(titles)):
        title = titles[i].strip()
        steps_list = re.findall(r"- (.+)", steps_raw[i]) if i < len(steps_raw) else []
        steps = "\n".join(f"{j+1}. {step}" for j, step in enumerate(steps_list))
        result = expected[i].strip() if i < len(expected) else ""
        rows.append({
            "Include": False,
            "Title": title,
            "Steps": steps,
            "Expected Result": result,
        })
    return pd.DataFrame(rows)

# --- STREAMLIT APP ---
st.set_page_config(page_title="QA Test Case Review Bot", layout="wide")
st.title("🤖 QA Test Case Review Bot")

# Layout: Sidebar for LLM config
with st.sidebar:
    st.header("🔧 LLM Settings")
    llm_choice = st.selectbox("Choose LLM", ["OpenAI", "Ollama", "Groq"])

    openai_key = ollama_endpoint = groq_endpoint = groq_key = groq_model = None
    if llm_choice == "OpenAI":
        openai_key = st.text_input("OpenAI API Key", type="password")
    elif llm_choice == "Ollama":
        ollama_endpoint = st.text_input("Ollama Endpoint (e.g., http://localhost:11434)")
        ollama_model = st.text_input("Ollama Model (e.g., llama2, mistral, tinyllama)")
    elif llm_choice == "Groq":
        groq_endpoint = st.text_input("Groq API Endpoint")
        groq_key = st.text_input("Groq API Key", type="password")
        groq_model = st.text_input("Groq Model (e.g., meta-llama/llama-4-scout-17b-16e-instruct)")

# Right panel
col1, col2 = st.columns([1, 3])
with col2:
    uploaded_file = st.file_uploader("📁 Upload Test Case Excel File", type=["xls", "xlsx"])
    user_story = st.text_area("✍️ Enter User Story")
    acceptance_criteria = st.text_area("✅ Enter Acceptance Criteria")

# Load uploaded test cases
original_df = pd.DataFrame()
if uploaded_file:
    try:
        original_df = pd.read_excel(uploaded_file)
        st.markdown("### 📋 Uploaded Test Cases")
        display_df = original_df.copy()
        display_df.index = display_df.index + 1
        st.dataframe(display_df)
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Initialize session
if "suggested_df" not in st.session_state:
    st.session_state.suggested_df = pd.DataFrame()

# Generate suggestions
if st.button("🔍 Review and Suggest Missing Test Cases") and not original_df.empty:
    if llm_choice == "OpenAI" and not openai_key:
        st.error("Please provide OpenAI API key.")
    elif llm_choice == "Ollama" and not ollama_endpoint:
        st.error("Please provide Ollama endpoint.")
    elif llm_choice == "Groq" and (not groq_endpoint or not groq_key or not groq_model):
        st.error("Please provide Groq endpoint, API key, and model.")
    else:
        with st.spinner(f"🔍 Analyzing using LLM ({llm_choice})..."):
            prompt = f"""
You are a QA expert. Given this user story and acceptance criteria:

User Story:
{user_story}

Acceptance Criteria:
{acceptance_criteria}

Here are existing test cases:
{original_df.to_string(index=False)}

Identify any missing test scenarios including negative and edge cases. Format:
Title: <title>
Steps:
- step 1
- step 2
Expected Result: <result>
            """
            try:
                if llm_choice == "OpenAI":
                    response = call_openai(openai_key, prompt)
                elif llm_choice == "Ollama":
                    response = call_ollama(ollama_endpoint, ollama_model, prompt)
                elif llm_choice == "Groq":
                    response = call_groq(groq_endpoint, groq_key, groq_model, prompt)

                st.session_state.suggested_df = parse_llm_response(response)

            except Exception as e:
                st.error(f"Error during LLM call: {str(e)}")

# Suggested test case display
if not st.session_state.suggested_df.empty:
    st.markdown("### 🧠 Suggested Test Cases (Edit + Approve to Include)")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ Select All"):
            st.session_state.suggested_df["Include"] = True
    with col2:
        if st.button("❌ Deselect All"):
            st.session_state.suggested_df["Include"] = False
    with col3:
        if st.button("➕ Add New Test Case"):
            st.session_state.suggested_df.loc[len(st.session_state.suggested_df)] = {
                "Include": False,
                "Title": "",
                "Steps": "",
                "Expected Result": ""
            }

    edited_df = st.data_editor(
        st.session_state.suggested_df,
        use_container_width=True,
        hide_index=True,
        column_order=["Include", "Title", "Steps", "Expected Result"],
        column_config={
            "Include": st.column_config.CheckboxColumn("Include")
        },
        key="suggested_editor"
    )
    st.session_state.suggested_df = edited_df

# Download final Excel
if not original_df.empty:
    selected_suggestions = pd.DataFrame(columns=["Title", "Steps", "Expected Result"])
    if (
            "suggested_df" in st.session_state
            and not st.session_state.suggested_df.empty
            and "Include" in st.session_state.suggested_df.columns
    ):
        selected_suggestions = st.session_state.suggested_df[
            st.session_state.suggested_df["Include"] == True
            ][["Title", "Steps", "Expected Result"]]

    final_df = pd.concat([original_df, selected_suggestions], ignore_index=True)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        final_df.to_excel(writer, index=False)
    output.seek(0)

    st.download_button(
        label="📥 Download Final Test Cases",
        data=output,
        file_name="final_test_cases.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
