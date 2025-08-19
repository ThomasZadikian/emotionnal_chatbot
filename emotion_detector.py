from transformers import pipeline, set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "microsoft/phi-3-mini-4k-instruct"

emotion_classifier = pipeline("text-classification",
                              model="j-hartmann/emotion-english-distilroberta-base")

sarcasm_detector = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-irony")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

response_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

def detect_emotion(text):
    if not text.strip():
        return {'label': 'neutral', 'score': 1.0}
    result = emotion_classifier(text)[0]
    return result

def detect_sarcasm(text):
        """
    Détecte si un texte est sarcastique en utilisant un modèle pré-entraîné.
    Args:
        text (str): Le texte à analyser.
    Returns:
        dict: Un dictionnaire contenant l'étiquette (sarcasme ou non) et son score.
    """
        if not text.strip(): 
             return {'label':'non-sarcastic', 'score': 1.0}
        result = sarcasm_detector(text)[0]
        if result['label'] in ['sarcasm', 'LABEL_1']:
             result['label'] = 'sarcasm'
        else: 
             result['label'] = 'not_sarcasm'
             
        return result

def analyse_text(text):
    """
    Analyse le texte pour détecter les émotions et le sarcasme.
    Args:
        text (str): Le texte à analyser.
    Returns:
        dict: Un dictionnaire contenant les résultats de l'analyse.
    """
    emotion_result = detect_emotion(text)
    sarcasm_result = detect_sarcasm(text)

    combined_result = {
         'emotion': emotion_result['label'],
         'emotion_score': emotion_result['score'], 
         'sarcasm': sarcasm_result['label'], 
         'sarcasm_score': sarcasm_result['score']
    }
    
    return combined_result

def generate_response(emotion, user_text, conversation_history=None):
    """
    Génère une réponse basée sur le contexte émotionnel en utilisant Phi-3.
    """
    prompt = f"Système : Tu es un chatbot empathique. Réponds avec bienveillance.\n"
    prompt += f"Contexte émotionnel détecté : {emotion}\n"
    prompt += f"Utilisateur : {user_text}\n"
    prompt += f"Assistant :"

    generated_text = response_generator(
        prompt,
        max_new_tokens=100,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )[0]['generated_text']

    response = generated_text.split("Assistant :")[-1].strip()
    
    return response

if __name__ == "__main__":
    print("\n" + "=" * 30)
    print("DÉMARRAGE DU CHATBOT (Émotionnel)")
    print("=" * 30)

    while True:
        user_input = input("Vous (Tapez 'quit' pour quitter) : ")

        if user_input.lower() == 'quit':
            print("Merci d'avoir discuté. Au revoir !")
            break
        
        analysis = analyse_text(user_input)
        emotion = analysis['emotion']
        sarcasm = analysis['sarcasm']

        generated_response = generate_response(emotion, user_input, conversation_history=None)
        
        print(f"Chatbot : {generated_response}")