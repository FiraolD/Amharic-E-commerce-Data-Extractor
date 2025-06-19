import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
from seqeval.metrics import f1_score


# Load your labeled CSV file
labeling_df = pd.read_csv('Data/sample_label.csv')

# Open output file for writing
with open('amharic_ner.conll', 'w', encoding='utf-8') as f:
    current_message = None
    for _, row in labeling_df.iterrows():
        if row['message_id'] != current_message:
            current_message = row['message_id']
            f.write('\n')  # New message, add blank line
        token = str(row['token']).strip()
        label = str(row['label']).strip()
        if token:
            f.write(f"{token}\t{label}\n")

def parse_conll(file_path):
    tokens, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        token_seq, label_seq = [], []
        for line in f:
            line = line.strip()
            if not line:
                if token_seq:
                    tokens.append(token_seq)
                    labels.append(label_seq)
                    token_seq, label_seq = [], []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    token, label = parts[0], parts[1]
                    token_seq.append(token)
                    label_seq.append(label)
        if token_seq:
            tokens.append(token_seq)
            labels.append(label_seq)
    return Dataset.from_dict({"tokens": tokens, "ner_tags": labels})

# Load dataset
dataset = parse_conll("amharic_ner.conll")


label2id = {
    "O": 0,
    "B-PRODUCT": 1,
    "I-PRODUCT": 2,
    "B-PRICE": 3,
    "I-PRICE": 4,
    "B-LOC": 5,
    "I-LOC": 6
}

id2label = {v: k for k, v in label2id.items()}

def encode_labels(example):
    return {"labels": [label2id[label] for label in example["ner_tags"]]}

dataset = dataset.map(encode_labels)


tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(label2id[label[word_idx]] if word_idx is not None else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)


model = AutoModelForTokenClassification.from_pretrained(
    "xlm-roberta-base", num_labels=len(label2id), id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="amharic_ner_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
)

trainer.train()


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "accuracy": accuracy_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }
    
    model.save_pretrained("./amharic_ner_model")
tokenizer.save_pretrained("./amharic_ner_model")