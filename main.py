import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Set Streamlit page config
st.set_page_config(page_title="Farmers AI")

# Load model and tokenizer
@st.cache_resource
def load_model():   
    tokenizer = T5Tokenizer.from_pretrained("CropSeek-LLM")
    model = T5ForConditionalGeneration.from_pretrained("CropSeek-LLM")
    return tokenizer, model

tokenizer, model = load_model()
model.eval()

# Title and description
st.title("🌾 Farmers AI")

# Updated default questions
default_questions = [
    "What is the best time to plant rice?",
    "How to manage pests in tomato crops?",
    "What are the signs of potassium deficiency in plants?",
    "Which irrigation method is best for sugarcane?"
]

# Initialize session state to track conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to get answer from the model
def get_answer(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Function to display chat bubbles
def chat_bubble(sender, message):
    if sender == "user":
        st.markdown(
            f"""
            <div style='text-align:right; background-color:#DCF8C6; padding:10px 15px; border-radius:15px; margin:10px 0; max-width:75%; float:right; clear:both; color:black;'>
                <b>You:</b> {message}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(
            f"""
            <div style='text-align:left; background-color:#F1F0F0; padding:10px 15px; border-radius:15px; margin:10px 0; max-width:75%; float:left; clear:both; color:black;'>
                <b>Bot:</b> {message}
            </div>
            """, unsafe_allow_html=True)

# Show previous messages
for msg in st.session_state.messages:
    chat_bubble(msg["sender"], msg["message"])

# Quick question buttons
if not st.session_state.messages:
    col1, col2 = st.columns(2)
    for i in range(2):
        if col1.button(default_questions[i]):
            user_question = default_questions[i]
            st.session_state.messages.append({"sender": "user", "message": user_question})
            bot_answer = get_answer("question: " + user_question)
            st.session_state.messages.append({"sender": "bot", "message": bot_answer})
            st.rerun()

        if col2.button(default_questions[i+2]):
            user_question = default_questions[i+2]
            st.session_state.messages.append({"sender": "user", "message": user_question})
            bot_answer = get_answer("question: " + user_question)
            st.session_state.messages.append({"sender": "bot", "message": bot_answer})
            st.rerun()

# User input form
with st.form(key="user_input_form", clear_on_submit=True):
    user_input = st.text_input("Ask your farming question here...")
    submit_button = st.form_submit_button("Send")

if submit_button and user_input:
    st.session_state.messages.append({"sender": "user", "message": user_input})
    bot_answer = get_answer("question: " + user_input)
    st.session_state.messages.append({"sender": "bot", "message": bot_answer})
    st.rerun()
