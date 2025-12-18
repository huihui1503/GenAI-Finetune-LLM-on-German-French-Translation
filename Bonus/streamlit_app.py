import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import os 
from peft import PeftModel
import re

os.environ["HF_TOKEN"] = "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"
test_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
system_prompt = "You are an assistant that translates German to French. You only reply in French and only give the answer without any explanations."
# 📌 Help to extract the response only.
def extract_between(text, start_pattern, end_pattern):
    match = re.search(f"{re.escape(start_pattern)}(.*?){re.escape(end_pattern)}", text, re.DOTALL)
    return match.group(1).strip() if match else ""

# 📌 Generate response according to prompt.
def generate_response(prompt, model, tokenizer):
  encoded_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
  model_inputs = encoded_input

  generated_ids = model.generate(**model_inputs,
                                  max_new_tokens=512,
                                  min_new_tokens=1,
                                  do_sample=False,
                                  pad_token_id=tokenizer.eos_token_id,
                                  eos_token_id=tokenizer.eos_token_id)

  decoded_output = tokenizer.batch_decode(generated_ids,)
  return extract_between(decoded_output[0],"<|start_header_id|>assistant<|end_header_id|>\n\n","<|eot_id|>")

@st.cache_resource
def load_fine_tuned_model(directory, model_name):
    model_base = AutoModelForCausalLM.from_pretrained(
        model_name,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_eos_token = False  # always False for inference
    )
    fine_tuned_model = PeftModel.from_pretrained(
        model_base,
        directory,
        force_download=True)
    print("Peft model loaded")
    return fine_tuned_model, tokenizer

adapter_directory = "/workspaces/translation-assistant-app/checkpoint-5080"

pretrained_model, tokenizer = load_fine_tuned_model(adapter_directory,"meta-llama/Llama-3.2-1B-Instruct")
st.title("Übersetzer")

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Schreiben Sie bitte"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = generate_response(test_prompt.format(system_prompt, prompt), pretrained_model, tokenizer)
    response = f"Übersetzung: {response}"
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})