import os
import time
import psutil
import numpy as np
import gradio as gr
from resemblyzer import VoiceEncoder, preprocess_wav

# Initialize the VoiceEncoder
encoder = VoiceEncoder()

def analyze_voice_similarity(audio_file1, audio_file2, progress=gr.Progress()):
    progress(0, desc="Starting analysis...")

    if not audio_file1 or not audio_file2:
        return "", "", "", "", "Please upload both audio files"

    start_time = time.time()
    process = psutil.Process(os.getpid())

    try:
        progress(0.2, desc="Processing first audio file...")
        wav1 = preprocess_wav(audio_file1)

        progress(0.4, desc="Processing second audio file...")
        wav2 = preprocess_wav(audio_file2)
    except Exception as e:
        return "", "", "", "", f"Error processing audio files: {str(e)}"

    progress(0.6, desc="Extracting voice embeddings...")
    embed1 = encoder.embed_utterance(wav1)
    embed2 = encoder.embed_utterance(wav2)

    progress(0.8, desc="Calculating similarity...")
    similarity = np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))

    result = "SAME PERSON" if similarity >= 0.80 else "DIFFERENT PEOPLE"

    memory_usage = process.memory_info().rss / (1024 * 1024)
    execution_time = time.time() - start_time

    progress(1.0, desc="Analysis complete!")

    return (
        f"{similarity:.4f}",
        result,
        f"{memory_usage:.2f} MB",
        f"{execution_time:.4f} seconds",
        ""
    )


with gr.Blocks(title="Voice Similarity Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Voice Similarity Analyzer")
    gr.Markdown(
        "Upload two audio files to check if they're from the same person. "
        "A similarity score >= 0.80 indicates the same speaker."
    )

    with gr.Row():
        with gr.Column():
            audio_input1 = gr.Audio(label="Voice Sample 1", type="filepath")
        with gr.Column():
            audio_input2 = gr.Audio(label="Voice Sample 2", type="filepath")

    analyze_button = gr.Button("Analyze Voice Similarity", variant="primary")

    error_message = gr.Markdown(visible=True)

    with gr.Group(visible=True):
        gr.Markdown("## Results")
        with gr.Row():
            with gr.Column():
                similarity_display = gr.Textbox(label="Similarity Score", value="")
            with gr.Column():
                conclusion_display = gr.Textbox(label="Conclusion", value="")
        with gr.Row():
            with gr.Column():
                memory_display = gr.Textbox(label="Memory Usage", value="")
            with gr.Column():
                time_display = gr.Textbox(label="Execution Time", value="")

    analyze_button.click(
        fn=analyze_voice_similarity,
        inputs=[audio_input1, audio_input2],
        outputs=[
            similarity_display,
            conclusion_display,
            memory_display,
            time_display,
            error_message,
        ],
    )

    gr.Markdown("""
    ## How It Works
    1. Upload two voice recordings
    2. Click "Analyze Voice Similarity"
    3. The app extracts voice embeddings using Resemblyzer
    4. Similarity is calculated using cosine similarity
    5. A score >= 0.80 indicates the same speaker
    """)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
