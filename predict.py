import torch
from transformers import pipeline

model_path = "emotion_classifier_results/checkpoint-1875"

print(f"Chargement du modèle depuis : {model_path}")

emotion_classifier = pipeline(
    "text-classification",
    model=model_path,
    top_k=None
)

def predict_emotions(text, threshold=0.3):
    """
    Prédit les émotions pour un texte donné, montre les top 3 scores bruts, 
    et filtre ensuite selon un seuil.
    """
    print(f"\nPrédiction pour le texte : '{text}'")
    
    predictions = emotion_classifier(text)
    results = predictions[0]
    
    all_emotions_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
    top_3_emotions = [(e['label'], f"{e['score']:.2%}") for e in all_emotions_sorted[:3]]
    print(f"Top 3 des prédictions (scores bruts) : {top_3_emotions}")

    detected_emotions = []
    for emotion in results:
        if emotion['score'] > threshold:
            formatted_score = f"{emotion['score']:.2%}"
            detected_emotions.append((emotion['label'], formatted_score))
            
    detected_emotions.sort(key=lambda x: x[1], reverse=True)
    
    return detected_emotions

# Test of prediction function
sentence_example = [
    "I am so happy that I get to work on this exciting project!",
    "This is really frustrating, the code is not working at all.",
    "I love the way you think about this problem, it's so insightful!",
]
for sentence in sentence_example:
    print(predict_emotions(sentence))
    print("\n--- Fin de la prédiction des émotions ---\n")


# Execution of prediction for user input
while True:
    sentence = input("Entrez une phrase pour détecter les émotions (ou 'exit' pour quitter) : ")
    if sentence.lower() == 'exit':
        break
    else:
        open('/logs/conversation_log.txt', 'a').write(f"Phrase: {sentence}\nÉmotions détectées: {emotions}\n\n")
        emotions = predict_emotions(sentence)