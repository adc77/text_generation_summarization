import streamlit as st
from model import get_models

gpt2_tokenizer, gpt2_model, bart_tokenizer, bart_model = get_models()

st.title("Text Generation and Summarization Tool")

option = st.selectbox("Choose Task", ["Text Generation", "Summarization"])

if option == "Text Generation":
    prompt = st.text_area("Enter your prompt:")
    if st.button("Generate Text"):
        inputs = gpt2_tokenizer(prompt, return_tensors="pt")
        outputs = gpt2_model.generate(inputs.input_ids, max_length=500)
        generated_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write(generated_text)
else:
    text = st.text_area("Enter text to summarize:")
    if st.button("Summarize"):
        inputs = bart_tokenizer(text, return_tensors="pt")
        summary_ids = bart_model.generate(inputs.input_ids, max_length=500)
        summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        st.write(summary)
