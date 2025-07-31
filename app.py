import streamlit as st
from transformers import pipeline, set_seed

# Load model only once
@st.cache_resource
def load_generator():
    generator = pipeline('text-generation', model='distilgpt2')
    set_seed(42)
    return generator

st.title("🧠 Offline Text Generator (GPT-2)")

prompt = st.text_input("Enter a prompt:")

if st.button("Generate"):
    generator = load_generator()
    output = generator(prompt, max_length=100, num_return_sequences=1)
    st.write("### Output:")
    st.success(output[0]['generated_text'])
