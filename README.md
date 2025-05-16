# NLP_assignment

# BioBERT Abbreviation & Long-Form Detector (Streamlit)

A lightweight Streamlit app that uses a fine-tuned BioBERT model to tag biomedical abbreviations and their long forms. Each interaction is logged to `streamlit_logs.jsonl`.

---

## Requirements

- Python 3.8+  
- (Optional) GPU + CUDA for faster inference  
- Create a file `requirements.txt` containing:
streamlit
torch
transformers

yaml
Copy
Edit

---

## Quickstart

1. **Clone the repo**  
 ```bash
 git clone https://github.com/your-username/your-repo.git
 cd your-repo
Set up a virtual environment

bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate.bat     # Windows
Install dependencies

bash
Copy
Edit
pip install --upgrade pip
pip install -r requirements.txt
Place your BioBERT checkpoint
Ensure the folder ./biobert_results/checkpoint-375 contains the model and tokenizer files.

Run the app

bash
Copy
Edit
streamlit run app.py
Browse to http://localhost:8501

Usage
Enter a biomedical sentence or phrase in the text box.

Click Predict to see token-level tags.

All inputs, predictions, and timestamps are appended to streamlit_logs.jsonl.

Configuration
Model path: Edit the model_path variable in app.py if your checkpoint lives elsewhere.

Log file name: Change LOG_FILE in app.py to customize the log filename.
