!pip install streamlit transformers pyngrok --quiet
!ngrok config add-authtoken "2zNSFPaA3Y12M4gs715JxAAO17w_2oFrXtGe6eJBshY8gHjtH"
print("Ngrok token added and packages installed.")

%%writefile app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource
def load_model():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def translate_code(source_code, source_lang, target_lang, tokenizer, model):
    prompt = f"Translate this code from {source_lang} to {target_lang}:\n{source_code}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=512,
        num_beams=5,
        early_stopping=True
    )
    translated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_code

def main():
    st.set_page_config(page_title="🔀 AI Code Translator", layout="wide")
    st.title("🔀 AI Code Translator (Python ↔ JavaScript)")
    st.markdown("Paste your code and translate between Python and JavaScript!")

    tokenizer, model = load_model()

    col1, col2 = st.columns(2)

    with col1:
        source_lang = st.selectbox("From:", ["Python", "JavaScript"])
        source_code = st.text_area(
            f"Enter {source_lang} code:",
            height=300,
            placeholder="def hello():\n    print('Hello World!')" if source_lang == "Python"
                        else "function hello() {\n  console.log('Hello World!');\n}"
        )

    with col2:
        target_lang = "JavaScript" if source_lang == "Python" else "Python"
        st.markdown(f"### ➡️ Translated {target_lang} Code")

        if st.button("Translate"):
            if not source_code.strip():
                st.warning("Please enter some code!")
            else:
                with st.spinner("Translating..."):
                    translated = translate_code(source_code, source_lang, target_lang, tokenizer, model)
                st.code(translated, language=target_lang.lower())

if __name__ == "__main__":
    main()


from pyngrok import ngrok
import threading
import time
import subprocess

def run_app():
    print("Launching Streamlit...")
    subprocess.run(["streamlit", "run", "app.py", "--server.port", "8501"])

thread = threading.Thread(target=run_app)
thread.start()

# Wait for Streamlit to start
time.sleep(60)

# Connect Ngrok
public_url = ngrok.connect(8501)
print("🚀 App is live at:", public_url)
