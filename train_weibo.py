import transformers
import torch
import pandas as pd
import numpy as np

from transformers import TrainingArguments, Trainer
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

model_name = "bert-base-chinese"
max_length = 128


TRAIN_FILE = "./data_ch/ren/data/train/news.csv"
TEST_FILE = "./data_ch/ren/data/test/news.csv"
OUTPUT_DIR = "./model_ch"

with open(TRAIN_FILE) as f:
    df_train = pd.read_csv(f)
with open(TEST_FILE) as f:
    df_test = pd.read_csv(f)

#remove missing data
df_train_clean = df_train.dropna()
df_test_clean = df_test.dropna()


titles_train = df_train_clean["Title"].to_list()
texts_train = df_train_clean["Report Content"].to_list()
labels_train = df_train_clean.label.to_list()

df_train_clean["content"] = df_train_clean["Title"]+df_train_clean["Report Content"] 
joined_content_train = df_train_clean.content.to_list()





#(train_titles,valid_titles,train_labels,rem_labels)=train_test_split(titles_train, labels_train, test_size=0.05)


titles_test = df_test_clean["Title"].to_list()
texts_test = df_test_clean["Report Content"].to_list()
labels_test = df_test_clean.label.to_list()

df_test_clean["content"] = df_test_clean["Title"]+df_test_clean["Report Content"] 
joined_content_test = df_test_clean.content.to_list()
(train_texts,valid_texts,train_labels,valid_labels)=train_test_split(texts_train, labels_train, test_size=0.05)

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
test_encodings = tokenizer(texts_test, truncation=True, padding=True, max_length=max_length)


train_dataset = Dataset(train_encodings, train_labels)
valid_dataset = Dataset(valid_encodings, valid_labels)
test_dataset = Dataset(test_encodings, labels_test)

#set up training
model=BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
training_args = TrainingArguments(
    num_train_epochs=2,
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
test_acc = compute_metrics((test_pred,labels_test))

print("train results")
print(train_acc)

print("valid results")
print(valid_acc)

print("test results")
print(test_acc)