import transformers
import torch
import pandas as pd
import numpy as np
import opencc
import json
import argparse

from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import TrainingArguments, Trainer



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




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str,help='"en" for English, "tw" for Taiwanese, "cn" for Chinese')

    parser.add_argument('input_file', type=argparse.FileType('r'),help='csv file with the news articles (must contain "title" column for English and "content" for the rest)')
    
    args = parser.parse_args()
    
    if args.src == 'en':
        model_path = "model_en/checkpoint-1000"  # Load trained model
        model_name = "bert-base-uncased"
        
        
        df = pd.read_csv(args.input_file,sep=";",usecols=['title'])
        clean_df = df.dropna()
        test = clean_df.title.to_list()
            
    else:
        model_path = "model_ch/checkpoint-1000"  # Load trained model
        model_name = "bert-base-chinese"
        df = pd.read_csv(args.input_file,sep=";",usecols=['content'])
        clean_df = df.dropna()
            
        if args.src == 'tw':
            converter = opencc.OpenCC('t2s.json')
            test_sim = clean_df.content.to_list()
            test = [converter.convert(x) for x in test_sim]
        else:
            test = clean_df.content.to_list()
    

    
    tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True) 
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2) # Define test trainer


    # Tokenize test data
    test_tokenized = tokenizer(test, padding=True, truncation=True, max_length=128) # Create torch dataset
    test_dataset = Dataset(test_tokenized)

    test_trainer = Trainer(model) # Make prediction
    raw_pred, _, _ = test_trainer.predict(test_dataset) # Preprocess raw predictions
    y_pred = np.argmax(raw_pred, axis=1)
    
    
    #get number of positive, negative and the ratio
    fake = np.count_nonzero(y_pred == 0)
    real = np.count_nonzero(y_pred == 1)
    
    ratio = fake/(fake+real)#not fake/real to avoid zero division
    results = {'fake':fake,'real':real,'ratio':ratio}
    file_name = "results_"+args.src+".json"
    with open(file_name,"w") as f:
        json.dump(results,f)