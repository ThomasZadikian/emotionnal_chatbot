import gradio as gr
from transformers import pipeline
import os

print("Loading the model... Please wait.")

model_path = "emotion_classifier_results/checkpoint-1875"

if not os.path.exists(model_path):
    raise OSError(f"Model path not found: {model_path}. Please verify the path.")

emotion_classifier = pipeline(
    "text-classification",
    model=model_path,
    top_k=None
)
print("Model loaded successfully! Launching the Gradio app...")


def predict_emotions(text):
    """
    This function takes a text string and returns a dictionary of
    emotion labels and their scores, formatted for Gradio.
    """
    predictions = emotion_classifier(text)[0]
    
    results_dict = {p['label']: p['score'] for p in predictions}
    
    return results_dict

iface = gr.Interface(
    fn=predict_emotions,
    inputs=gr.Textbox(lines=3, placeholder="Enter a sentence here..."),
    outputs=gr.Label(num_top_classes=5, label="Detected Emotions"),
    title="🤖 Text Emotion Analyzer",
    description="A model fine-tuned to detect 28 nuanced emotions. This project was built with the help of Doctorat, a research assistant.",
    examples=[
        ["I am so happy that I get to work on this exciting project!"],
        ["This is really frustrating, the code is not working at all."],
        ["Oh, great. Another meeting. Just what I needed."]
    ]
)

iface.launch()