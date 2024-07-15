import streamlit as st
from yt_dlp import YoutubeDL
from haystack.nodes import PromptNode, PromptModel
from haystack.nodes.audio import WhisperTranscriber
from haystack.pipelines import Pipeline
from model_add import LlamaCPPInvocationLayer
import torch
import os
import requests

st.set_page_config(layout="wide")

# Ensure that we are using GPU if available
use_gpu = torch.cuda.is_available()

# URL to download the model from
model_url = "https://huggingface.co/TheBloke/Llama-2-7B-32K-Instruct-GGUF/resolve/main/llama-2-7b-32k-instruct.Q4_K_S.gguf"

# Path to save the downloaded model
model_path = "/content/llama-2-7b-32k-instruct.Q4_K_S.gguf"

# Function to download the model
def download_model(url, path):
    if not os.path.exists(path):
        st.info(f"Downloading model from {url}...")
        response = requests.get(url, stream=True)
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        st.success("Model downloaded successfully!")
    else:
        st.info("Model already exists, skipping download.")

# Function to download the video
def download_video(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': '/content/%(title)s.%(ext)s',
    }
    with YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(result)

def initialize_model(model_path):
    return PromptModel(
        model_name_or_path=model_path,
        invocation_layer_class=LlamaCPPInvocationLayer,
        use_gpu=use_gpu
    )

def initialize_prompt_node(model):
    summary_prompt = "deepset/summarization"
    return PromptNode(model_name_or_path=model, default_prompt_template=summary_prompt, use_gpu=use_gpu)

def transcribe_audio(file_path, prompt_node):
    whisper = WhisperTranscriber()
    pipeline = Pipeline()
    pipeline.add_node(component=whisper, name="whisper", inputs=["File"])
    pipeline.add_node(component=prompt_node, name="prompt", inputs=["whisper"])
    output = pipeline.run(file_paths=[file_path])
    return output

def main():
    # Set the title and background color
    st.title("YouTube Video Summarizer 🎥")
    st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)
    st.subheader('Built with the Llama 2 🦙, Haystack, Streamlit and ❤️')
    st.markdown('<style>h3{color: pink;  text-align: center;}</style>', unsafe_allow_html=True)

    # Expander for app details
    with st.expander("About the App"):
        st.write("This app allows you to summarize while watching a YouTube video.")
        st.write("Enter a YouTube URL in the input box below and click 'Submit' to start. This app is built by AI enthusiasts.")

    # Input for YouTube URL
    url = st.text_input("Enter YouTube URL")
    if st.button("Submit"):
        with st.spinner('Downloading video...'):
            video_file = download_video(url)
            st.success('Video downloaded!')

        with st.spinner('Downloading model...'):
            download_model(model_url, model_path)

        with st.spinner('Initializing model...'):
            model = initialize_model(model_path)
            prompt_node = initialize_prompt_node(model)
            st.success('Model initialized!')

        with st.spinner('Transcribing and summarizing audio...'):
            summary = transcribe_audio(video_file, prompt_node)
            st.success('Transcription and summarization complete!')
            st.write(summary)

if __name__ == "__main__":
    main()
