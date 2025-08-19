import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score, precision_score, recall_score
import evaluate

print("Chargement des données...")
go_emotions_dataset = load_dataset("go_emotions", "raw")

if 'validation' not in go_emotions_dataset:
    print("Split de validation manquant. Création à partir du split d'entraînement (90/10)...")
    train_validation_split = go_emotions_dataset['train'].train_test_split(test_size=0.1, seed=42)
    
    go_emotions_dataset['train'] = train_validation_split['train']
    go_emotions_dataset['validation'] = train_validation_split['test']

print("Structure des splits du dataset :", go_emotions_dataset)

print("Prétraitement des données...")
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

meta_columns = {'text', 'id', 'author', 'subreddit', 'link_id', 'parent_id', 'created_utc', 'rater_id', 'example_very_unclear'}
emotion_names = [col for col in go_emotions_dataset['train'].features if col not in meta_columns]

def preprocess_function(examples):
    # Le reste de la fonction est inchangé...
    tokenized_inputs = tokenizer(examples['text'], truncation=True, padding=True)
    labels = []
    for i in range(len(examples['text'])):
        label = [0.0] * len(emotion_names)
        for idx, emotion in enumerate(emotion_names):
            if examples[emotion][i] == 1:
                label[idx] = 1.0
        labels.append(label)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# It is better to use GoogleColab or a similar environment to handle large datasets
# Do not use only CPU for training, as it will be very slow
# If you are using a local machine, ensure you have enough RAM and GPU support
# Here we will use a smaller subset for demonstration purposes
# The beggining of a pertinent result are with 5000 & 2000
train_dataset = go_emotions_dataset['train'].select(range(2000))
eval_dataset = go_emotions_dataset['validation'].select(range(500))

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

print("Chargement du modèle pré-entraîné...")
id2label = {idx: label for idx, label in enumerate(emotion_names)}
label2id = {label: idx for idx, label in enumerate(emotion_names)}

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, 
    num_labels=len(emotion_names),
    problem_type="multi_label_classification",
    id2label=id2label,
    label2id=label2id
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= 0.5)] = 1
    y_true = labels
    
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        'f1_weighted': f1_weighted,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
    }
    return metrics
print("Configuration des arguments d'entraînement...")
training_args = TrainingArguments(
    output_dir="emotion_classifier_results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("Lancement de l'entraînement...")
trainer.train()

print("Entraînement terminé !")