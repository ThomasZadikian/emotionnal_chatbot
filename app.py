# app.py
import gradio as gr
from transformers import pipeline
import torch
import os

print("Loading models... This is the final step and may take a moment.")

emotion_classifier_path = "emotion_classifier_results/checkpoint-1875" 
if not os.path.exists(emotion_classifier_path):
    raise OSError(f"Emotion classifier not found at {emotion_classifier_path}")
emotion_classifier = pipeline("text-classification", model=emotion_classifier_path, top_k=None)

chatbot = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16, device_map="auto")

print("Models loaded! Launching the Gradio app...")

def generate_empathetic_response(user_input):
    """
    Analyzes emotion and generates an empathetic response using a modern chat model.
    """
    emotions = emotion_classifier(user_input)[0]
    emotions_dict = {p['label']: p['score'] for p in emotions}
    primary_emotion = max(emotions_dict, key=emotions_dict.get)
    
    messages = [
        {"role": "system", "content": f"You are an empathetic and caring chatbot. Your primary goal is to respond to the user's feelings. Acknowledge their emotion, which is '{primary_emotion}', and offer a supportive and understanding message. Do not be generic and always tell me what emotion you are using."},
        {"role": "user", "content": user_input},
    ]
    
    prompt = chatbot.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    outputs = chatbot(
        prompt,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    
    response_text = outputs[0]["generated_text"]
    chatbot_response = response_text[len(prompt):].strip()
    
    return emotions_dict, chatbot_response

iface = gr.Interface(
    fn=generate_empathetic_response,
    inputs=gr.Textbox(lines=3, placeholder="How are you feeling today?"),
    outputs=[
        gr.Label(num_top_classes=5, label="Detected Emotions"),
        gr.Textbox(label="Empathetic Chatbot Response")
    ],
    title="🤖 Empathetic AI Chatbot (ft. TinyLlama)",
    description="This chatbot first analyzes the emotion in your text, then uses a modern generative model to formulate a context-aware response.",
    examples=[["I finally finished my big project, I'm so relieved!"], ["I can't believe the server crashed again."]]
)

iface.launch()