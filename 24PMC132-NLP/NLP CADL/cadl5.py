# ==============================================================================
# 1. SETUP: INSTALL AND IMPORT LIBRARIES (WITH LOGGING CONTROL)
# ==============================================================================
# Install necessary libraries silently
!pip install transformers torch sentencepiece gradio -q

import gradio as gr
from transformers import pipeline
import textwrap
import logging

# Suppress informational messages and warnings from the transformers library
logging.getLogger("transformers").setLevel(logging.ERROR)


# ==============================================================================
# 2. LOAD THE PRE-TRAINED LLM (SILENTLY)
# ==============================================================================
# Load the pre-trained summarization pipeline. All download progress and messages are now hidden.
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# ==============================================================================
# 3. DEFINE THE SUMMARIZATION FUNCTION
# ==============================================================================
# This is the core function that will be connected to our user interface.
def summarize_text(article, min_length, max_length):
    """
    Takes an article and length parameters, returns a summarized version.
    """
    if not article.strip():
        return "Please provide some text to summarize."

    try:
        summary_list = summarizer(
            article,
            min_length=int(min_length),
            max_length=int(max_length),
            do_sample=False,
            truncation=True
        )
        summary_text = summary_list[0]['summary_text']
        return textwrap.fill(summary_text, width=100)
    except Exception as e:
        return f"An error occurred during summarization: {e}"


# ==============================================================================
# 4. BUILD THE INTERACTIVE USER INTERFACE WITH GRADIO
# ==============================================================================
with gr.Blocks(theme=gr.themes.Soft(), title="AI Text Summarizer") as demo:
    gr.Markdown("# 📝 AI-Powered Text Summarizer")
    gr.Markdown("Paste your long article or text below and get a concise summary in seconds. Adjust the sliders to control the summary length.")

    with gr.Row():
        article_input = gr.Textbox(lines=20, label="Input Text / Article", placeholder="Paste your full article here...")
        summary_output = gr.Textbox(lines=20, label="Generated Summary", interactive=False)

    with gr.Row():
        min_len_slider = gr.Slider(minimum=30, maximum=200, value=60, step=10, label="Minimum Summary Length")
        max_len_slider = gr.Slider(minimum=100, maximum=500, value=250, step=10, label="Maximum Summary Length")

    summarize_button = gr.Button("Generate Summary ✨", variant="primary")

    gr.Markdown("### Try it with an example:")
    gr.Examples(
        examples=[
            [
            """
            Quantum computing is a revolutionary type of computation that leverages the principles of quantum mechanics to process information in fundamentally new ways. Unlike classical computers, which use bits to represent information as either a 0 or a 1, quantum computers use qubits. A qubit can exist in a superposition of both 0 and 1 simultaneously, and multiple qubits can be linked together through a phenomenon called quantum entanglement. This allows quantum computers to perform a vast number of calculations at once. The potential applications are immense, ranging from developing new medicines and materials by simulating molecular structures, to breaking current cryptographic codes and creating new, more secure ones. However, building and maintaining stable quantum computers is incredibly challenging. Qubits are extremely fragile and sensitive to their environment, a problem known as decoherence, which can destroy the quantum state and introduce errors into the computation. Researchers around the world are working to overcome these obstacles, developing better error-correction techniques and more robust qubit designs to unlock the full potential of this transformative technology.
            """,
            60,
            150
            ]
        ],
        inputs=[article_input, min_len_slider, max_len_slider]
    )

    summarize_button.click(
        fn=summarize_text,
        inputs=[article_input, min_len_slider, max_len_slider],
        outputs=summary_output
    )

# ==============================================================================
# 5. LAUNCH THE APPLICATION (CLEANLY)
# ==============================================================================
# quiet=True: Hides the "Running on local URL..." message.
# share=True: Required for a public link in Colab, prevents the auto-detection message.
demo.launch(quiet=True, share=True)