import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from scripts.model import t5_predict_station
import os
import warnings
warnings.filterwarnings("ignore")

# Streamlit app for detecting Bangkok Metro Station Names from text, with NER and T5 Model


@st.cache_resource
def load_t5_model():
    '''Load the T5 model trained for Bangkok Metro Station NER. This return model and its tokenizer'''
    t5_model_path = os.path.join("models", "deep_learning")
    model = T5ForConditionalGeneration.from_pretrained(t5_model_path)
    tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
    return model, tokenizer


def run():
    '''Run the Streamlit app for detecting Bangkok Metro Station Name from text.'''
    # Streamlit UI
    st.title("Bangkok Metro Station NER")

    st.markdown('**This is a demo application for identifying metro station names from a given text, for 8 skytrain stations in BTS Silom Line (Dark Green Line without extension:).**')
    hide_footer_style = """<style>.reportview-container .main footer {visibility: hidden;}"""
    st.markdown(hide_footer_style, unsafe_allow_html=True)

    # Load the T5 model
    model, tokenizer = load_t5_model()

    # Input text box
    st.subheader("Enter text with station name(s)")
    user_input = st.text_area("Enter your sentence with BTS station(s), for up to two stations:", height=100)

    # Predict button
    if st.button("Predict"):
        if user_input.strip():
            # Get prediction from T5 model
            prediction = t5_predict_station(model, tokenizer, user_input)
            # Display the detection result
            st.subheader("Detected Station(s):")
            if prediction == "":
                st.success("No station detected.")
            else:
                st.success(prediction)
        else:
            st.warning("Please enter a sentence first.")


if __name__ == "__main__":
    run()