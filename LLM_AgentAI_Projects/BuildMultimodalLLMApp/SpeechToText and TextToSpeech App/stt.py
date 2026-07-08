import os
import uuid
import tempfile

import gradio as gr
import whisper
from gtts import gTTS

from langchain_ibm import ChatWatsonx
from langchain_core.messages import HumanMessage, SystemMessage


# -----------------------------
# IBM watsonx configuration
# -----------------------------
# Set these as environment variables before running:
#
# Windows PowerShell:
# setx WATSONX_API_KEY "your_api_key"
# setx WATSONX_PROJECT_ID "your_project_id"
# setx WATSONX_URL "https://us-south.ml.cloud.ibm.com"
#
# Linux/Mac:
# export WATSONX_API_KEY="your_api_key"
# export WATSONX_PROJECT_ID="your_project_id"
# export WATSONX_URL="https://us-south.ml.cloud.ibm.com"

WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")


# -----------------------------
# Load Whisper model once
# -----------------------------
# Options: tiny, base, small, medium, large
# Use "base" for normal laptop testing.
whisper_model = whisper.load_model("base")


# -----------------------------
# Load IBM watsonx model
# -----------------------------
def get_ibm_llm():
    if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
        return None

    llm = ChatWatsonx(
        model_id="ibm/granite-3-8b-instruct",
        url=WATSONX_URL,
        apikey=WATSONX_API_KEY,
        project_id=WATSONX_PROJECT_ID,
        params={
            "temperature": 0.4,
            "max_new_tokens": 300,
        },
    )

    return llm


ibm_llm = get_ibm_llm()


# -----------------------------
# Speech-to-text using Whisper
# -----------------------------
def speech_to_text(audio_path):
    if audio_path is None:
        return "No audio provided."

    result = whisper_model.transcribe(audio_path)
    transcript = result["text"].strip()

    return transcript


# -----------------------------
# IBM LangChain response
# -----------------------------
def generate_response_with_ibm(user_text):
    if not user_text.strip():
        return "Please provide some text or audio first."

    if ibm_llm is None:
        return (
            "IBM watsonx is not configured. "
            "Please set WATSONX_API_KEY, WATSONX_PROJECT_ID, and WATSONX_URL."
        )

    messages = [
        SystemMessage(
            content=(
                "You are a helpful voice assistant. "
                "Answer clearly and concisely."
            )
        ),
        HumanMessage(content=user_text),
    ]

    response = ibm_llm.invoke(messages)
    return response.content


# -----------------------------
# Text-to-speech using gTTS
# -----------------------------
def text_to_speech(text, language):
    if not text.strip():
        return None

    output_path = os.path.join(
        tempfile.gettempdir(),
        f"tts_output_{uuid.uuid4().hex}.mp3"
    )

    tts = gTTS(text=text, lang=language)
    tts.save(output_path)

    return output_path


# -----------------------------
# Full pipeline:
# Audio -> Whisper STT -> IBM response -> gTTS audio
# -----------------------------
def voice_chat(audio_path, language):
    transcript = speech_to_text(audio_path)

    if transcript == "No audio provided.":
        return transcript, "", None

    bot_response = generate_response_with_ibm(transcript)
    audio_output = text_to_speech(bot_response, language)

    return transcript, bot_response, audio_output


# -----------------------------
# Text-only TTS
# -----------------------------
def simple_tts(text, language):
    audio_output = text_to_speech(text, language)
    return audio_output


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="STT + TTS Voice Assistant") as demo:
    gr.Markdown("# STT + TTS Voice Assistant")
    gr.Markdown(
        "Upload or record audio. Whisper transcribes it, IBM watsonx generates a response, "
        "and gTTS converts the response to speech."
    )

    with gr.Tab("Voice Chat"):
        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Record or upload audio"
        )

        language_input = gr.Dropdown(
            choices=[
                ("English", "en"),
                ("German", "de"),
                ("Hindi", "hi"),
                ("French", "fr"),
                ("Spanish", "es"),
            ],
            value="en",
            label="TTS Language"
        )

        run_button = gr.Button("Transcribe and Respond")

        transcript_output = gr.Textbox(
            label="Whisper Transcript",
            lines=4
        )

        response_output = gr.Textbox(
            label="IBM watsonx Response",
            lines=6
        )

        audio_output = gr.Audio(
            label="Generated TTS Audio",
            type="filepath"
        )

        run_button.click(
            fn=voice_chat,
            inputs=[audio_input, language_input],
            outputs=[transcript_output, response_output, audio_output]
        )

    with gr.Tab("Only Text to Speech"):
        text_input = gr.Textbox(
            label="Enter text",
            lines=5,
            placeholder="Type something to convert into speech..."
        )

        tts_language = gr.Dropdown(
            choices=[
                ("English", "en"),
                ("German", "de"),
                ("Hindi", "hi"),
                ("French", "fr"),
                ("Spanish", "es"),
            ],
            value="en",
            label="TTS Language"
        )

        tts_button = gr.Button("Generate Speech")

        tts_audio_output = gr.Audio(
            label="TTS Output",
            type="filepath"
        )

        tts_button.click(
            fn=simple_tts,
            inputs=[text_input, tts_language],
            outputs=tts_audio_output
        )


if __name__ == "__main__":
    demo.launch()