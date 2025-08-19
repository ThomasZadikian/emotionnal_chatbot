import gradio as gr
from transformers import pipeline
import torch
import os

print("Loading models... This may take a moment.")

emotion_classifier = pipeline("text-classification", model="emotion_classifier_results/checkpoint-1875", top_k=None)
chatbot = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16, device_map="auto")

print("Models loaded! Launching the Gradio app...")


def generate_response(user_input, persona):
    """
    Analyzes emotion and generates a response based on the selected persona.
    """
    emotions = emotion_classifier(user_input)[0]
    emotions_dict = {p['label']: p['score'] for p in emotions}
    primary_emotion = max(emotions_dict, key=emotions_dict.get)
    
    if persona == "Empathetic Persona":
        system_prompt = f"You are an empathetic chatbot. Acknowledge that the user is feeling '{primary_emotion}' and offer a supportive message."
    else:
        system_prompt = "You are a neutral, factual chatbot. Respond directly to the user's statement without acknowledging emotions."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    
    prompt = chatbot.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    outputs = chatbot(prompt, max_new_tokens=64, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    
    response_text = outputs[0]["generated_text"]
    chatbot_response = response_text[len(prompt):].strip()
    
    return emotions_dict, chatbot_response


iface = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(lines=3, placeholder="How are you feeling today?"),
        gr.Radio(["Empathetic Persona", "Neutral Persona"], label="Select a Persona", value="Empathetic Persona")
    ],
    outputs=[
        gr.Label(num_top_classes=5, label="Detected Emotions"),
        gr.Textbox(label="Chatbot Response")
    ],
    title="🤖 Persona-Driven AI Chatbot",
    description="Select a persona and enter a sentence. The chatbot's response style will change based on your selection.",
    examples=[
        ["I'm so angry, my project just crashed!", "Empathetic Persona"],
        ["I'm so angry, my project just crashed!", "Neutral Persona"],
        ["This is the best news I've had all week!", "Empathetic Persona"],
    ]
)

iface.launch()