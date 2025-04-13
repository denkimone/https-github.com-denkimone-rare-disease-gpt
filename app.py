import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import json
import os
import base64

# --- Page setup ---
st.set_page_config(page_title="Rare Disease GPT", page_icon="🎗")

# --- Optional background image ---
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None

b64 = get_base64_image("Picture1.png")
if b64:
    st.markdown(f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{b64}") no-repeat center center fixed;
        background-size: cover;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- Load model ---
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def load_qa_pipeline():
    # your code to load a QA pipeline, like from Hugging Face
    from transformers import pipeline
    return pipeline("question-answering")
qa = load_qa_pipeline()

# --- Load dataset ---
@st.cache_data
def load_dataset():
    try:
        with open("rare_disease_qa.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("🚫 Dataset not found: rare_disease_qa.json")
        return []

dataset = load_dataset()

# --- Find exact matching context from dataset ---
def retrieve_context(query, dataset):
    query_lower = query.lower()
    for item in dataset:
        if query_lower in item["question"].lower():
            return item["question"], item["answer"]
    return None, None

# --- UI and Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False

def render_conversation():
    html = '<div class="chat-container">'
    for msg in st.session_state.messages:
        alignment = "flex-start" if msg["sender"] == "bot" else "flex-end"
        html += f'<div class="chat-message" style="justify-content:{alignment};">'
        html += f'<div class="chat-bubble"><p>{msg["content"]}</p></div></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# --- Landing page ---
if not st.session_state.show_chat:
    st.title("🎗 Rare Disease GPT")
    st.markdown("Ask questions about Rare Medical Conditions")
    if st.button("🚀 Start Chat"):
        st.session_state.show_chat = True
        st.rerun()
else:
    st.title("🎗 Rare Disease GPT")
    render_conversation()

    user_input = st.text_input("Ask questions about Rare Medical Conditions:", key="user_input")

    if st.button("ASK") and user_input.strip():
        matched_q, matched_a = None, None  # <-- Add this line
        st.session_state.messages.append({"sender": "user", "content": user_input})

        matched_q, matched_a = retrieve_context(user_input, dataset)
        
        context = ""
        if matched_q and matched_a:
            st.session_state.messages.append({"sender": "bot", "content": f"{matched_a}"})
        else:
            st.session_state.messages.append({"sender": "bot", "content": "❌ Sorry, I couldn't find a matching question. Try rephrasing it."})

        st.rerun()

        context = f"You have access to this related Q&A:\nQ: {matched_q}\nA: {matched_a}\n\n"
        prompt = (
        f"{context}"
        "You are a knowledgeable medical assistant specializing in rare diseases. "
        "Provide a thorough, detailed, and educational answer to the user's question. "
        "Include definitions, symptoms, causes, complications, and recommended treatments or management steps. "
        "Write in multiple sentences or bullet points for clarity.\n\n"
        f"User's Question: {user_input}\n\nAnswer:"
    )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = model.generate(**inputs, max_length=512, temperature=0.7, top_p=0.95)
        bot_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        st.session_state.messages.append({"sender": "bot", "content": bot_reply})
        render_conversation()
