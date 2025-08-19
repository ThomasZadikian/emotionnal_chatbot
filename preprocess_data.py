from datasets import load_dataset
from transformers import AutoTokenizer

print("Chargement du dataset et du tokenizer...")
go_emotions_dataset = load_dataset("go_emotions", "raw")
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

meta_columns = {'text', 'id', 'author', 'subreddit', 'link_id', 'parent_id', 'created_utc', 'rater_id', 'example_very_unclear'}
emotion_names = [col for col in go_emotions_dataset['train'].features if col not in meta_columns]

def preprocess_function(examples):
    tokenized_inputs = tokenizer(examples['text'], truncation=True)
    
    labels = []
    for i in range(len(examples['text'])):
        label = [0.0] * len(emotion_names)
        for idx, emotion in enumerate(emotion_names):
            if examples[emotion][i] == 1:
                label[idx] = 1.0
        labels.append(label)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


print("Application de la fonction de prétraitement au dataset...")
tokenized_dataset = go_emotions_dataset.map(preprocess_function, batched=True)


print("\nExemple après prétraitement :")
print(tokenized_dataset['train'][0])