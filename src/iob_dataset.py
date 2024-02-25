import pdb
import traceback
import pickle

import random
import pandas as pd
import torch
from datasets import load_dataset
import tqdm
import platform
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import datetime
import os
# Setting up the TensorBoard callback

# model.fit(..., callbacks=[tensorboard_callback])
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import seqeval

from seqeval.metrics import classification_report

#from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader

from transformers import AdamW, BertTokenizer, BertForTokenClassification

import json
from jinja2 import Template


class IOBSyntheticDataset(Dataset):
    def __init__(self, tokenizer,number_of_records, nutrients_table_path, json_data_source, max_len,internal_state=None,is_validation_set=False,augmented_context=[],output_path_name=None):
        self.len = int(number_of_records)
        self.load_data_pickle = self.load_data_pickle
        self.internal_state = internal_state
        if internal_state == None:

            self.generator = ProductLabelDataGenerator(nutrients_table_path,  json_data_source, augmented_context=augmented_context, is_validation_set=is_validation_set, total_lines=self.len, output_path_name=output_path_name)
        print('got after ******')
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augmented_context = augmented_context

        self.label2id = {
                'O': 0,
                'B-MACRONUTRIENTS': 1,
                'I-MACRONUTRIENTS': 2
            }
    

    def save_data_pickle(self, data, filename):
        with open(filename, 'wb') as file:
            pickle.dump(data, file)

    def load_data_pickle(self, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)


    def save_state(self):
        print('Saving state file.')
        self.generator.save()
        
    def tokenize_and_preserve_labels(self, sentence, text_labels, tokenizer):
        tokenized_sentence = []
        labels = []

        #sentence = sentence.strip()

        for word, label in zip(sentence, text_labels):

            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)

            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)

            # Add the same label to the new list of labels `n_subwords` times
            labels.extend([label] * n_subwords)

        return tokenized_sentence, labels

    # Function to tokenize and align labels
    # def tokenize_and_align_labels(self, examples):
    #     tokenized_inputs = self.tokenizer(examples['text'], truncation=True, is_split_into_words=True)
    #     labels = []
    #     for i, label in enumerate(examples['labels']):
    #         word_ids = tokenized_inputs.word_ids(batch_index=i)
    #         previous_word_idx = None
    #         label_ids = []
    #         for word_idx in word_ids:
    #             if word_idx is None:
    #                 label_ids.append(-100)
    #             elif word_idx != previous_word_idx:
    #                 label_ids.append(label2id[label[word_idx]])
    #             else:
    #                 label_ids.append(label2id[label[word_idx]])
    #             previous_word_idx = word_idx
    #         labels.append(label_ids)
    #     tokenized_inputs["labels"] = labels
    #     return tokenized_inputs


    def __getitem__(self, index):
        
        try:
            if self.internal_state != None:
                return self.internal_state[index]
                
            # step 1: tokenize (and adapt corresponding labels)
            #iob, compiled_label, context_iob_only = self.generator.get_data_sequence(index)
            #iob_df = pd.DataFrame(iob, columns=['words', 'tokens'])
            compiled_label = self.generator.data.iloc[index]['product_label_raw']
            iob_structure = self.generator._generate_iob(compiled_label)
            # if random.random() > .5:
            #     iob_df = pd.DataFrame(context_iob_only, columns=['words', 'tokens'])

            sentence, word_labels = zip(*iob_structure)

           # sentence = iob['words']
           # word_labels = iob['tokens']
            tokenized_sentence, labels = self.tokenize_and_preserve_labels(sentence, word_labels, self.tokenizer)
            
            # step 2: add special tokens (and corresponding labels)
            tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"] # add special tokens
            labels.insert(0, "O") # add outside label for [CLS] token
            labels.insert(-1, "O") # add outside label for [SEP] token

            # step 3: truncating/padding
            maxlen = self.max_len

            if (len(tokenized_sentence) > maxlen):
                # truncate
                tokenized_sentence = tokenized_sentence[:maxlen]
                labels = labels[:maxlen]
            else:
                # pad
                tokenized_sentence = tokenized_sentence + ['[PAD]'for _ in range(maxlen - len(tokenized_sentence))]
                labels = labels + ["O" for _ in range(maxlen - len(labels))]

            # step 4: obtain the attention mask
            attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]
            
            # step 5: convert tokens to input ids
            ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)

            label_ids = [self.label2id[label.upper()] for label in labels]
            # the following line is deprecated
            #label_ids = [label if label != 0 else -100 for label in label_ids]
            # tokenized_sentence['ids'] =  torch.tensor(ids, dtype=torch.long)
            # tokenized_sentence['mask'] =  torch.tensor(attn_mask, dtype=torch.long)
            # tokenized_sentence['targets'] =  torch.tensor(label_ids, dtype=torch.long)
            # return tokenized_sentence

            return {
                'input_ids': torch.tensor(ids, dtype=torch.long),
                'attention_mask': torch.tensor(attn_mask, dtype=torch.long),
                #'token_type_ids': torch.tensor(token_ids, dtype=torch.long),
                'labels': torch.tensor(label_ids, dtype=torch.long)
            } 
        except Exception as e:
            #pdb.set_trace()
            print(f'Error processing index {index} {e}')
            #traceback.print_exc()
            #raise(e)
            return self.__getitem__(index -1)
    

    def __len__(self):
        return int(self.len)
    
    
    def get_category(self,category_id):
        if int(category_id) in self.nutrients.index:
            return self.nutrients.loc[category_id]['Category'].upper().rstrip()
        return "OTHER"
    
    
    def create_product_label_data(self, data):
        
        product_label = {
            "description": data.get("description", ""),
            "brand_owner": data.get("brandOwner", ""),
            "ingredients": data.get("ingredients", ""),
            "label_nutrients": data.get("labelNutrients", ""),
            "market_country": data.get("marketCountry", ""),
            "serving_size": f"{data.get('servingSize', '')} {data.get('servingSizeUnit', '')}",
            "nutrients": []
        }
        
        # create dataframe with 'description' and 'brand_owner' as columns
        
        for nutrient in data.get("foodNutrients", []):
            nutrient_info = {
                "name":  nutrient.get("nutrient", {}).get("name", ""),
                "quantity": nutrient.get("amount", ""),
                "unit": nutrient.get("nutrient", {}).get("unitName", ""),
                "Category": self.get_category(nutrient.get("nutrient", {}).get("id", 1149))
            }
            product_label["nutrients"].append(nutrient_info)
        # return as dataframe
        
        return product_label


def is_macos():
    return platform.system() == 'Darwin'

class Trainer:
    def __init__(self):
        pass
    
    def inference(self, tokenizer, model, sentence,max_len=512):
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if is_macos() else "cpu"))

        inputs = tokenizer(sentence, padding='max_length', truncation=True, max_length=max_len, return_tensors="pt")

        # move to gpu
        ids = inputs["input_ids"].to(device)
        mask = inputs["attention_mask"].to(device)
        # forward pass
        outputs = model(ids, mask)
        logits = outputs[0]

        active_logits = logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level

        tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
        token_predictions = [id2label[i] for i in flattened_predictions.cpu().numpy()]
        wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)

        word_level_predictions = []
        for pair in wp_preds:
            if (pair[0].startswith(" ##")) or (pair[0] in ['[CLS]', '[SEP]', '[PAD]']):
                # skip prediction
                continue
            else:
                word_level_predictions.append(pair[1])

        # we join tokens, if they are not special ones
        str_rep = " ".join([t[0] for t in wp_preds if t[0] not in ['[CLS]', '[SEP]', '[PAD]']]).replace(" ##", "")
        print(str_rep)
        print(word_level_predictions)
    
    def valid(self, model, testing_loader, id2label):
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if is_macos() else "cpu"))

        # put model in evaluation mode
        model.eval()
        
        eval_loss, eval_accuracy = 0, 0
        nb_eval_examples, nb_eval_steps = 0, 0
        eval_preds, eval_labels = [], []
        
        with torch.no_grad():
            for idx, batch in enumerate(testing_loader):
                
                ids = batch['ids'].to(device, dtype = torch.long)
                mask = batch['mask'].to(device, dtype = torch.long)
                targets = batch['targets'].to(device, dtype = torch.long)
                
                outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
                loss, eval_logits = outputs.loss, outputs.logits
                
                eval_loss += loss.item()

                nb_eval_steps += 1
                nb_eval_examples += targets.size(0)
            
                if idx % 100==0:
                    loss_step = eval_loss/nb_eval_steps
                    print(f"Validation loss per 100 evaluation steps: {loss_step}")
                
                # compute evaluation accuracy
                flattened_targets = targets.view(-1) # shape (batch_size * seq_len,)
                active_logits = eval_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
                flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
                # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
                active_accuracy = mask.view(-1) == 1 # active accuracy is also of shape (batch_size * seq_len,)
                targets = torch.masked_select(flattened_targets, active_accuracy)
                predictions = torch.masked_select(flattened_predictions, active_accuracy)
                
                eval_labels.extend(targets)
                eval_preds.extend(predictions)
                
                tmp_eval_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
                eval_accuracy += tmp_eval_accuracy
        
        #print(eval_labels)
        #print(eval_preds)

        labels = [id2label[id.item()] for id in eval_labels]
        predictions = [id2label[id.item()] for id in eval_preds]

        #print(labels)
        #print(predictions)
        
        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_steps
        print(f"Validation Loss: {eval_loss}")
        print(f"Validation Accuracy: {eval_accuracy}")
        print(classification_report([labels], [predictions]))
        return labels, predictions
    
    # Defining the training function on the 80% of the dataset for tuning the bert model
    def _train(self, epoch, model, training_loader,output_path_name="/", __save=None,train_dataset=None):
        MAX_LEN = 128
        TRAIN_BATCH_SIZE = 4
        VALID_BATCH_SIZE = 2
        EPOCHS = 1
        LEARNING_RATE = 1e-05
        MAX_GRAD_NORM = 10
        
        import torch.optim as optim

        # --> optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        # optimizer = optim.Adam(model.parameters(), lr=0.001)
        #swa_model = AveragedModel(model)
        #scheduler = CosineAnnealingLR(optimizer, T_max=100)
   
        #optimizer = torch.optim.Adam(params=model.parameters(), weight_decay=0.2, lr=LEARNING_RATE)
        optimizer = AdamW(params=model.parameters(), weight_decay=0.8, lr=LEARNING_RATE)
        
        
        tr_loss, tr_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        tr_preds, tr_labels = [], []
        # put model in training mode
        model.train()
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if is_macos() else "cpu"))
        print(f'...moving model to device {device}')
        model.to(device)
        
        writer = SummaryWriter(f'{output_path_name}/{epoch}_exp')
        total_count = len(training_loader)
        current = 0
        save_count = 0
        for idx, batch in tqdm.tqdm(enumerate(training_loader), total=len(training_loader), desc="Training"):
            current = current + 1
            percent_complete = (current + 1) / total_count * 100

            # save on every %5 completed
            if current == 1500:
                print('saving on percentaget completion..')
                __save(idx)
                train_dataset.save_state()
                writer.flush()
                current = 0
            # print hello every 200
            save_count = save_count + 1
            if save_count == 200:
                train_dataset.save_state()
                save_count = 0
            ids = batch['ids'].to(device, dtype = torch.long)
            mask = batch['mask'].to(device, dtype = torch.long)
            targets = batch['targets'].to(device, dtype = torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
            loss, tr_logits = outputs.loss, outputs.logits
            writer.add_scalar('Loss/train', loss, epoch)
            tr_loss += loss.item()

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)
            
            if idx % 100 == 0:
                loss_step = tr_loss/nb_tr_steps
                print()
                print(f"Training loss per 100 training steps: {loss_step}")

            # compute training accuracy
            flattened_targets = targets.view(-1) # shape (batch_size * seq_len,)
            active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
            active_accuracy = mask.view(-1) == 1 # active accuracy is also of shape (batch_size * seq_len,)
            targets = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)
            
            tr_preds.extend(predictions)
            tr_labels.extend(targets)
            
            tmp_tr_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
            tr_accuracy += tmp_tr_accuracy
        
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=MAX_GRAD_NORM
            )
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        writer.close()
        epoch_loss = tr_loss / nb_tr_steps
        tr_accuracy = tr_accuracy / nb_tr_steps
        print(f"Training loss epoch: {epoch_loss}")
        print(f"Training accuracy epoch: {tr_accuracy}")
    
    def train(self,output_path,exp_name, context=[], model_path='bert-base-uncased',save_every_epoch=False,epochs=1, max_len=512, batch_size=2,valid_batch_size=2, number_of_records=5000, nutrients_table_path=None, json_data_source=None):
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if is_macos() else "cpu"))

        #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        # need to keep state of records alreadyn trained on...
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        output_path_name = f'{output_path}/{exp_name}'
        
        train_dataset = IOBSyntheticDataset(**{
            "max_len": max_len,
            "augmented_context": context,
            "number_of_records": number_of_records * 0.8,
            "tokenizer": tokenizer,
            "nutrients_table_path": nutrients_table_path,
            "json_data_source": json_data_source,
            
            "output_path_name": output_path_name
        })       
        
        test_dataset  = IOBSyntheticDataset(**{
            "max_len": max_len,
            "augmented_context": context,
            "number_of_records": number_of_records * 0.2,
            "tokenizer": tokenizer,
            "nutrients_table_path": nutrients_table_path,
            "json_data_source": json_data_source,
            "is_validation_set": True,
            
            "output_path_name": output_path_name
        })       
     
        
        label2id = {k: v for v, k in enumerate(train_dataset.label2id)}
        id2label = {v: k for v, k in enumerate(train_dataset.label2id)}
        
        
        training_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        
        model = BertForTokenClassification.from_pretrained(model_path, 
                                        num_labels=len(id2label),
                                        id2label=id2label,
                                        label2id=label2id)
        
        def __save(percent):
            try:
                
                save_path = os.path.join(output_path_name, f'model_{percent}')
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                

            except Exception as e:
                print(f'Error saving model{e}')
                pass
                
        try:
            epoch_state_file = os.path.join(output_path_name, f'epoch_state.csv')
            if not os.path.exists(epoch_state_file):
                # create epoch state file
                
                epoch_state = []
                for epoch in range(epochs):
                    epoch_state.append({'epoch': epoch, 'is_trained': False})
                epoch_state_df = pd.DataFrame(epoch_state, columns=['epoch', 'is_trained'])
                
                epoch_state_df.to_csv(epoch_state_file, index=False)
            epoch_state_df = pd.read_csv(epoch_state_file)    
            epochs_proposed = epochs
            epochs = epochs - epoch_state_df['is_trained'].sum()  
            print(f'Epochs proposed {epochs_proposed} and epochs to train {epochs}')
            for epoch in tqdm.tqdm(range(epochs), total=epochs, desc=f'Training {exp_name}'):
                self._train(epoch, model, training_loader, output_path_name, __save, train_dataset)
                train_dataset.save_state()
                epoch_state_df.loc[epoch_state_df['epoch'] == epoch, 'is_trained'] = True
                epoch_state_df.to_csv(epoch_state_file, index=False)
                
                if save_every_epoch:
                    save_path = os.path.join(output_path_name, 'model')
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    try:
                        model_eval = BertForTokenClassification.from_pretrained(save_path)
                        model_eval.to(device)
                        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                        self.valid(model_eval, test_loader, id2label)
                    except Exception as e:
                        print(f'Error loading model {e}')
                        pass
        except Exception as e:
            print(f'Error training {e}')
            pass
                
        

            
        
if '__main__' == __name__:
    
    from datasets import load_dataset
    dataset = load_dataset("yelp_review_full")
    
    trainer_arguments = {
        "model_path": "/content/drive/MyDrive/ML/ner/v1/model_15",
        "exp_name": "v1",
        "context": dataset['train']['text'],
        "output_path": "/content/drive/MyDrive/ML/ner", 
        # TPU-GPU (16
        # TPU (8)
        # V-100 (16)
        # A-100 (52)
        "batch_size": 48,
        "valid_batch_size": 2,
        "number_of_records": 20000,
        "max_len": 512,
        "save_every_epoch": True,
        "epochs": 4,
        "nutrients_table_path": "/content/food_ingredients_ner/src/data/nutrient.csv",
        "json_data_source": "/content/drive/MyDrive/ML/ner/brandedDownload.json/brandedDownload.json"
    }
    
    
    trainer = Trainer()
    trainer.train(**trainer_arguments)

    pass
    
    