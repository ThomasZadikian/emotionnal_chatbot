from datasets import load_dataset

print("Chargement du dataset GoEmotions...")
go_emotions_dataset = load_dataset("go_emotions", "raw")

meta_columns = {'text', 'id', 'author', 'subreddit', 'link_id', 'parent_id', 'created_utc', 'rater_id', 'example_very_unclear'}

emotion_names = [col for col in go_emotions_dataset['train'].features if col not in meta_columns]

print(f"\nDécouverte de {len(emotion_names)} colonnes d'émotions. Par exemple : {emotion_names[:5]}")

example = go_emotions_dataset['train'][0]
print(f"\nTexte de l'exemple : '{example['text']}'")

decoded_labels = []
for emotion in emotion_names:
    if example[emotion] == 1:
        decoded_labels.append(emotion)
        
print(f"Labels décodés de l'exemple : {decoded_labels}")