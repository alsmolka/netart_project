import transformers
import torch
import pandas as pd
import numpy as np

from transformers import TrainingArguments, Trainer
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

model_name = "bert-base-uncased"
max_length = 128


TRAIN_FILE = "./data_eng/train.csv"
OUTPUT_DIR = "./saved_model"

with open(TRAIN_FILE) as f:
    df = pd.read_csv(f)

#remove missing data
df_clean = df.dropna()

texts = df_clean.title.to_list()
labels = df_clean.label.to_list()

(train_texts,rem_texts,train_labels,rem_labels)=train_test_split(texts, labels, test_size=0.2)
(valid_texts,test_texts,valid_labels,test_labels)=train_test_split(rem_texts, rem_labels, test_size=0.5)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    
#tokenize and prep data
tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)


train_dataset = Dataset(train_encodings, train_labels)
valid_dataset = Dataset(valid_encodings, valid_labels)
test_dataset = Dataset(test_encodings, test_labels)

#set up training
model=BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
training_args = TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=16,  
    weight_decay=0.01,              
    load_best_model_at_end=True,
    logging_steps=500,
    evaluation_strategy="steps",
    output_dir = OUTPUT_DIR
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
trainer.train()

#evaluate
train_pred, _, _ = trainer.predict(train_dataset)
train_acc = compute_metrics((train_pred,train_labels))

valid_pred, _, _ = trainer.predict(valid_dataset)
valid_acc = compute_metrics((valid_pred,valid_labels))

test_pred, _, _ = trainer.predict(test_dataset)
test_acc = compute_metrics((test_pred,test_labels))

print("train results")
print(train_acc)

print("valid results")
print(valid_acc)

print("test results")
print(test_acc)