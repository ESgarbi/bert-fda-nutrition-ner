from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import pandas as pd


import ast

model_path = 'model_v2.2'

df = pd.read_parquet('./synthetic_sample_v2.2.parquet')




# infer string into list... maybe there is a better way of doing this
df['ner_literal'] = df['ner'].apply(lambda x: ast.literal_eval(x))


tags = []
for tag in df['ner_literal']:
    for t in tag:
        tags.append(t[1])

unique_tags = list(set(tags))

unique_tags = sorted(unique_tags)

unique_tags = sorted(unique_tags, reverse=True)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}

id2tag = {id: tag for tag, id in tag2id.items()}

print(tag2id)
print(id2tag)


tag2id = {'O': 0, 'I-VITAMINS': 1, 'I-STIMULANTS': 2, 'I-PROXIMATES': 3, 'I-PROTEIN': 4, 'I-PROBIOTICS': 5, 'I-MINERALS': 6, 'I-LIPIDS': 7, 'I-FLAVORING': 8, 'I-ENZYMES': 9, 'I-EMULSIFIERS': 10, 'I-DIETARYFIBER': 11, 'I-COLORANTS': 12, 'I-CARBOHYDRATES': 13, 'I-ANTIOXIDANTS': 14, 'I-ALCOHOLS': 15, 'I-ADDITIVES': 16, 'I-ACIDS': 17, 'B-VITAMINS': 18, 'B-STIMULANTS': 19, 'B-PROXIMATES': 20, 'B-PROTEIN': 21, 'B-PROBIOTICS': 22, 'B-MINERALS': 23, 'B-LIPIDS': 24, 'B-FLAVORING': 25, 'B-ENZYMES': 26, 'B-EMULSIFIERS': 27, 'B-DIETARYFIBER': 28, 'B-COLORANTS': 29, 'B-CARBOHYDRATES': 30, 'B-ANTIOXIDANTS': 31, 'B-ALCOHOLS': 32, 'B-ADDITIVES': 33, 'B-ACIDS': 34}

id2tag = {0: 'O', 1: 'I-VITAMINS', 2: 'I-STIMULANTS', 3: 'I-PROXIMATES', 4: 'I-PROTEIN', 5: 'I-PROBIOTICS', 6: 'I-MINERALS', 7: 'I-LIPIDS', 8: 'I-FLAVORING', 9: 'I-ENZYMES', 10: 'I-EMULSIFIERS', 11: 'I-DIETARYFIBER', 12: 'I-COLORANTS', 13: 'I-CARBOHYDRATES', 14: 'I-ANTIOXIDANTS', 15: 'I-ALCOHOLS', 16: 'I-ADDITIVES', 17: 'I-ACIDS', 18: 'B-VITAMINS', 19: 'B-STIMULANTS', 20: 'B-PROXIMATES', 21: 'B-PROTEIN', 22: 'B-PROBIOTICS', 23: 'B-MINERALS', 24: 'B-LIPIDS', 25: 'B-FLAVORING', 26: 'B-ENZYMES', 27: 'B-EMULSIFIERS', 28: 'B-DIETARYFIBER', 29: 'B-COLORANTS', 30: 'B-CARBOHYDRATES', 31: 'B-ANTIOXIDANTS', 32: 'B-ALCOHOLS', 33: 'B-ADDITIVES', 34: 'B-ACIDS'}



def extract_texts_and_tags(data):
    texts = [doc[0] for doc in data]
    tags = [doc[1] for doc in data]
    return texts, tags

# Initialize the tokenizer
tokenizer = BertTokenizerFast.from_pretrained(model_path)

def encode_tags(tags, encodings):
    encoded_labels = []
    for doc_idx, doc in enumerate(tags):
        word_ids = encodings.word_ids(batch_index=doc_idx)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(tag2id[doc[word_idx]])
            else:
                label_ids.append(tag2id[doc[word_idx]])
            previous_word_idx = word_idx
        
        encoded_labels.append(label_ids)
    
    return encoded_labels




class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.device = torch.device('mps')# if torch.cuda.is_available() else torch.device('cpu')

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).to(self.device)
        return item

    def __len__(self):
        return len(self.labels)



texts_list = []
tags_list = []
for index, row in df.iterrows():

    texts, tags = extract_texts_and_tags(row['ner_literal'])  
    texts_list.append(texts)
    tags_list.append(tags)


# Split the dataset into training and validation sets
train_texts, val_texts, train_tags, val_tags = train_test_split(texts_list, tags_list, test_size=.2)

train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)


# Encode the tags
train_labels = encode_tags(train_tags, train_encodings)
val_labels = encode_tags(val_tags, val_encodings)



train_dataset = NERDataset(train_encodings, train_labels)
val_dataset = NERDataset(val_encodings, val_labels)



device = torch.device('mps')# if torch.cuda.is_available() else torch.device('cpu')
model = BertForTokenClassification.from_pretrained(model_path, num_labels=len(unique_tags))
model.to(device)

training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=3,              
    per_device_train_batch_size=64,  
    per_device_eval_batch_size=16,   
   # warmup_steps=500,               
    weight_decay=0.05,               
    logging_dir='./logs',            
    logging_steps=50,
    evaluation_strategy="steps",
    save_steps=400,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    
)

trainer.train()

# Save the model
save_model_path = "model_v2.3"
model.save_pretrained(save_model_path)
tokenizer.save_pretrained(save_model_path)
