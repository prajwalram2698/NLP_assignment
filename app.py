import streamlit as st
import torch
from transformers import BertTokenizerFast, AutoModelForTokenClassification
import datetime
import json
import os

# 1. Setup logging
LOG_FILE = "streamlit_logs.jsonl"
if not os.path.exists(LOG_FILE):
    open(LOG_FILE, "w").close()

def log_interaction(user_text, predictions):
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "input_text": user_text,
        "predictions": [{"token": t, "tag": tag} for t, tag in predictions]
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

# 2. Load BioBERT once
@st.cache_resource(show_spinner=True)
def load_biobert_model():
    model_path = "./biobert_results/checkpoint-375"
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

tokenizer, model = load_biobert_model()
labels = ['O', 'B-AC', 'B-LF', 'I-LF']

# 3. Prediction function
def get_predictions(text):
    words = text.strip().split()
    encoded = tokenizer(words, is_split_into_words=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**encoded).logits
    preds = torch.argmax(logits, dim=2).squeeze().tolist()
    word_indices = encoded.word_ids()
    aligned = []
    prev_idx = None
    for i, idx in enumerate(word_indices):
        if idx is None:
            continue
        if idx != prev_idx:
            aligned.append((words[idx], labels[preds[i]]))
        prev_idx = idx
    return aligned

# 4. Streamlit UI
st.title("BioBERT-based Abbreviation & Long-Form Detection")
user_text = st.text_area("Enter a biomedical sentence or phrase:")

if st.button("Predict"):
    if not user_text.strip():
        st.warning("Please enter some text before predicting.")
    else:
        preds = get_predictions(user_text)
        # Display
        st.write("### Token-level Predictions")
        for tok, tag in preds:
            st.markdown(f"- **{tok}**: {tag}")
        # Log
        log_interaction(user_text, preds)
        st.success("Interaction logged.")
