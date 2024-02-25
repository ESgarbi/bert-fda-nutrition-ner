import pdb
import traceback
import threading
import pickle
import json
import concurrent.futures
from jinja2 import Template
import random
import pandas as pd
import torch
from datasets import load_dataset
import tqdm
import platform
import datetime
import os
from transformers import BertTokenizer


class IngredientItem():
    def __init__(self, tag, values):
        self.tag = tag
        self.values = values

        
    def __str__(self):
        return f'{self.tag} {self.values}'

    def __repr__(self):
        return f'{self.tag} {self.values}'

    def __eq__(self, other):
        return self.tag == other.tag and self.values == other.values

    def to_json(self):
        return json.dumps(self.__dict__)

    
    
    @staticmethod
    def from_json(json_data):
        data = json.loads(json_data)
        return IngredientItem(data)

class NutrientItem():
    def __init__(self, row):
        self.name = row['name']
        if ',' in self.name:
            self.name = self.name.split(',')[0].strip()
            
        #self.name = self.name.replace('-', ' ').replace('total:', '').replace('Total', '').replace('total', '').replace('by', '').replace('difference', '')

        self.unit = row['unit']
        self.amount = row['amount']
        self.tag = row['category']
        if 'vitamin' in self.name.lower():
            self.tag = 'VITAMINS'
        self.tag = self.tag.upper().rstrip()
        
    def __str__(self):
        return f'{self.name} {self.unit} {self.amount} {self.tag}'

    def __repr__(self):
        return f'{self.name} {self.unit} {self.amount} {self.tag}'

    def __eq__(self, other):
        return self.name == other.name and self.unit == other.unit and self.amount == other.amount and self.tag == other.tag

    def to_json(self):
        return json.dumps(self.__dict__)

    
    
    @staticmethod
    def from_json(json_data):
        data = json.loads(json_data)
        return NutrientItem(data)

class FDARecord():
    def __init__(self, row):
        self.row = row
        self.labels = set()
        self.description = row['description']
        self.brandOwner = row['brandOwner']
        self.gtinUpc = row['gtinUpc']
        self.servingSize = row['servingSize']
        self.servingSizeUnit = row['servingSizeUnit']
        self.brandedFoodCategory = row['brandedFoodCategory']
        self.publicationDate = row['publicationDate']
        if row['ingredients'] is not None:
            self.ingredients = self.build_ingredients(json.loads(row['ingredients']))
        else:
            self.ingredients = []  # or any other default value     
        

        self.nutrients = None
        if row['nutrients'] is not None:
            self.nutrients = self.build_nutrients(json.loads(row['nutrients']))
        else:
            self.nutrients = []  # or any other default value     
                
        
        
    def get_tagged_ingredients(self):
        
        ingredients_pairs = []
        
        for item in self.ingredients:
            for value in item.values:
                self.labels.add(item.tag)
                # if value is list
                if isinstance(value, list):
                    for v in value:
                        ingredients_pairs.append(f' <{item.tag}> {v} </{item.tag}> ')
                else:
                    ingredients_pairs.append(f' <{item.tag}> {value} </{item.tag}> ')
         # shuffle ingredients_pairs
        random.shuffle(ingredients_pairs)       
        return '  ,  '.join(ingredients_pairs)
        
    def build_ingredients(self, row):
        ingredients = []
        for item in row:
            values = row[item]
            self.labels.add(item)
            ingredients.append(IngredientItem(item, values))
            
        return ingredients
    def build_nutrients(self, row):
        nutrients = []
        for item in row:
            nutrients.append(NutrientItem(item))
            
        return nutrients
        
    def __str__(self):
        return f'{self.name} {self.unit}'

    def __repr__(self):
        return f'{self.name} {self.unit}'

    def __eq__(self, other):
        return self.name == other.name and self.unit == other.unit 

    def to_json(self):
        return json.dumps(self.__dict__)
    
    @staticmethod
    def build_data(parquet_file):
        records =[]
        df = pd.read_parquet(parquet_file)
        for index, row in df.iterrows():
            try:
                record = FDARecord(row)
                records.append(record)
            except Exception as e:

                pass
        return list(records)



# This class will generate synthetic data for training the NER model
class IOBTrainingDataGenerator():
    def __init__(self, parquet_file, json_fda_path, nutrients_table_path, tokenizer, max_seq_length, output_dir, training_file_size=10000, overwrite_cache=False):
        self.tokenizer = tokenizer
        self.records = FDARecord.build_data(parquet_file)
        self.training_file_size = training_file_size
        self.json_fda_path = json_fda_path
        self.max_len = 512
        self.index_file_current_count = 0
        self.__lock = threading.Lock()
        self.output_dir = output_dir
        self.exogenous_context = self.get_context()
        self.overwrite_cache = overwrite_cache
        self.templates = [
                    '''{{ product_info.description }}
                {{ product_info['packageWeight'] }} 
                {{ ingredient_label() }} {{ product_info.get_tagged_ingredients() }} 
                {{ nutritional_label() }}
                {% for nutrient in product_info.nutrients %}
                <{{ nutrient.tag }}> {{ nutrient.name }} </{{ nutrient.tag }}> {{ nutrient.amount }} 
                Nutrition per 100 grams: {{ get_one_hundred_grams(product_info.servingSize, nutrient.amount) }} {{ nutrient.unitName }}
                {% endfor %}''',
                
                    '''Product Description: {{ product_info.description }}
                Net Weight: {{ product_info['packageWeight'] }} 
                Ingredients: {{ ingredient_label() }} {{ product_info.get_tagged_ingredients() }} 
                Nutrition Facts:
                {% for nutrient in product_info.nutrients %}
                <{{ nutrient.tag }}> {{ nutrient.name }} </{{ nutrient.tag }}> {{ nutrient.amount }} 
                Serving Size Nutrition (100g): {{ get_one_hundred_grams(product_info.servingSize, nutrient.amount) }} {{ nutrient.unitName }}
                {% endfor %}''',
                
                    '''{{ product_info.description }}
                Weight: {{ product_info['packageWeight'] }} 
                Ingredients List: {{ ingredient_label() }} {{ product_info.get_tagged_ingredients() }} 
                Nutritional Information:
                {% for nutrient in product_info.nutrients %}
                <{{ nutrient.tag }}> {{ nutrient.name }} </{{ nutrient.tag }}> {{ nutrient.amount }} 
                Per 100g: {{ get_one_hundred_grams(product_info.servingSize, nutrient.amount) }} {{ nutrient.unitName }}
                {% endfor %}''',
                
                
                    '''Description: {{ product_info.description }}
                Package Weight: {{ product_info['packageWeight'] }} 
                List of Ingredients: {{ ingredient_label() }} {{ product_info.get_tagged_ingredients() }} 
                Nutritional Facts:
                {% for nutrient in product_info.nutrients %}
                <{{ nutrient.tag }}> {{ nutrient.name }} </{{ nutrient.tag }}> {{ nutrient.amount }} 
                Nutrient Content (100g): {{ get_one_hundred_grams(product_info.servingSize, nutrient.amount) }} {{ nutrient.unitName }}
                {% endfor %}''',
                
                
                    '''{{ product_info.description }}
                Total Weight: {{ product_info['packageWeight'] }} 
                Ingredients: {{ ingredient_label() }} {{ product_info.get_tagged_ingredients() }} 
                Nutritional Details:
                {% for nutrient in product_info.nutrients %}
                <{{ nutrient.tag }}> {{ nutrient.name }} </{{ nutrient.tag }}> {{ nutrient.amount }} 
                Nutrition per 100 grams: {{ get_one_hundred_grams(product_info.servingSize, nutrient.amount) }} {{ nutrient.unitName }}
                {% endfor %}''',
                
                    '''Product Description: {{ product_info.description }}
                Weight Specification: {{ product_info['packageWeight'] }} 
                Ingredient Information: {{ ingredient_label() }} {{ product_info.get_tagged_ingredients() }} 
                Nutrition Label:
                {% for nutrient in product_info.nutrients %}
                <{{ nutrient.tag }}> {{ nutrient.name }} </{{ nutrient.tag }}> {{ nutrient.amount }} 
                Nutritional Value (100g): {{ get_one_hundred_grams(product_info.servingSize, nutrient.amount) }} {{ nutrient.unitName }}
                {% endfor %}''',
                
                
                    '''{{ product_info.description }}
                Pack Weight: {{ product_info['packageWeight'] }} 
                Included Ingredients: {{ ingredient_label() }} {{ product_info.get_tagged_ingredients() }} 
                Nutrition Facts Label:
                {% for nutrient in product_info.nutrients %}
                <{{ nutrient.tag }}> {{ nutrient.name }} </{{ nutrient.tag }}> {{ nutrient.amount }} 
                100g Nutrition Info: {{ get_one_hundred_grams(product_info.servingSize, nutrient.amount) }} {{ nutrient.unitName }}
                {% endfor %}''',
                
                
                    '''Detailed Description: {{ product_info.description }}
                Weight of Package: {{ product_info['packageWeight'] }} 
                Ingredients Composition: {{ ingredient_label() }} {{ product_info.get_tagged_ingredients() }} 
                Nutritional Values:
                {% for nutrient in product_info.nutrients %}
                <{{ nutrient.tag }}> {{ nutrient.name }} </{{ nutrient.tag }}> {{ nutrient.amount }} 
                Nutrient Profile per 100g: {{ get_one_hundred_grams(product_info.servingSize, nutrient.amount) }} {{ nutrient.unitName }}
                {% endfor %}''',
                
                
                    '''{{ product_info.description }}
                Gross Weight: {{ product_info['packageWeight'] }} 
                Full Ingredients: {{ ingredient_label() }} {{ product_info.get_tagged_ingredients() }} 
                Nutrition Information:
                {% for nutrient in product_info.nutrients %}
                <{{ nutrient.tag }}> {{ nutrient.name }} </{{ nutrient.tag }}> {{ nutrient.amount }} 
                100g Nutritional Analysis: {{ get_one_hundred_grams(product_info.servingSize, nutrient.amount) }} {{ nutrient.unitName }}
                {% endfor %}''',
                
                
                    '''Product Overview: {{ product_info.description }}
                Package Net Weight: {{ product_info['packageWeight'] }} 
                Ingredients Breakdown: {{ ingredient_label() }} {{ product_info.get_tagged_ingredients() }} 
                Nutritional Fact Sheet:
                {% for nutrient in product_info.nutrients %}
                <{{ nutrient.tag }}> {{ nutrient.name }} </{{ nutrient.tag }}> {{ nutrient.amount }} 
                Nutritional Content (Per 100g): {{ get_one_hundred_grams(product_info.servingSize, nutrient.amount) }} {{ nutrient.unitName }}
                {% endfor %}'''
                ]
        
        self.label2id = {
                "O": "0",
                "B-ACIDITYREGULATORS": 1,
                "I-ACIDITYREGULATORS": 2,
                "B-ACIDS": 3,
                "I-ACIDS": 4,
                "B-ADDITIVES": 5,
                "I-ADDITIVES": 6,
                "B-ALCOHOLS": 7,
                "I-ALCOHOLS": 8,
                "B-AMINOACIDS": 9,
                "I-AMINOACIDS": 10,
                "B-CARBOHYDRATES": 11,
                "I-CARBOHYDRATES": 12,
                "B-CAROTENOIDS": 13,
                "I-CAROTENOIDS": 14,
                "B-COLORANTS": 15,
                "I-COLORANTS": 16,
                "B-FIBER": 17,
                "I-FIBER": 18,
                "B-FLAVONOIDS": 19,
                "I-FLAVONOIDS": 20,
                "B-FLAVORINGS": 21,
                "I-FLAVORINGS": 22,
                "B-LIPIDS": 23,
                "I-LIPIDS": 24,
                "B-MACRONUTRIENTS": 25,
                "I-MACRONUTRIENTS": 26,
                "B-MINERALS": 27,
                "I-MINERALS": 28,
                "B-ORGANICCOMPOUNDS": 29,
                "I-ORGANICCOMPOUNDS": 30,
                "B-PHYTOCHEMICALS": 31,
                "I-PHYTOCHEMICALS": 32,
                "B-PRESERVATIVES": 33,
                "I-PRESERVATIVES": 34,
                "B-PROTEINS": 35,
                "I-PROTEINS": 36,
                "B-PROXIMATES": 37,
                "I-PROXIMATES": 38,
                "B-STABILIZERS": 39,
                "I-STABILIZERS": 40,
                "B-STIMULANTS": 41,
                "I-STIMULANTS": 42,
                "B-SUGARS": 43,
                "I-SUGARS": 44,
                "B-VITAMINS": 45,
                "I-VITAMINS": 46,
                "B-WATER": 47,
                "I-WATER": 48
                }

        
    def get_context(self):
        dataset = load_dataset("yelp_review_full")
        return dataset['train']['text']
    
   
    def process_lines(self, chunk, num_samples):
        global __lock
        lines = []
        index = 0
        for row in tqdm.tqdm(chunk, total=len(chunk), desc='Processing lines...'):
            #print progress at 100
          
            try:

                data_label = self._generate_label_0(row)
                index += 1
                
                iob, weights = self._generate_iob(self.augument(data_label))
                
                if iob != None:
                    lines.append({"iob": iob, "weights": weights})  # assuming these are defined elsewhere
                #lines.append((self.augument(data_label))
                
            except Exception as e:
                pass
        
        self.save_state(lines)
        return lines

    def process_in_threads(self, num_threads):
        # Split the file into chunks
        
        chunks = [list() for _ in range(num_threads)]
        
        index = 0
        for row in self.records:
            chunks[index % num_threads].append(row)
            index += 1
            
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Schedule the processing of each chunk
            future_to_chunk = {executor.submit(self.process_lines, chunk, len(self.records)): chunk for chunk in chunks}
            for future in concurrent.futures.as_completed(future_to_chunk):
                results.extend(future.result())

        return results

    def augument(self, compiled_label):
        text_context = random.choice(self.exogenous_context)
        # choose position between 1 and 4
        position = random.randint(1, 4)

        if position == 1:
            #insert into the first paragraph of the text_context
            compiled_label = compiled_label + '\n' + text_context
        elif position == 2:    
            #insert into the second paragraph of the text_context
            text_context = text_context.split('\n')
            text_context.insert(1, compiled_label)
            compiled_label = '\n'.join(text_context)
        elif position == 3:
            #insert into the third paragraph of the text_context
            text_context = text_context.split('\n')
            text_context.insert(2, compiled_label)
            compiled_label = '\n'.join(text_context)
            
        elif position == 4:
            #insert into the third paragraph of the text_context
            text_context = text_context.split('\n')
            text_context.insert(2, compiled_label)
            compiled_label = '\n'.join(text_context)
                
        return compiled_label
        
    def _generate_label_0(self, product_info):      
        # product_info['description'] = product_info.get('description', 'No description available')
        # product_info['brandOwner'] = product_info.get('brandOwner', 'Unknown brand owner')
        # product_info['ingredients'] = product_info.get('ingredients', 'Ingredients not available')
        # product_info['marketCountry'] = product_info.get('marketCountry', 'Unknown market country')
        # product_info['servingSize'] = product_info.get('servingSize', 'Serving size not specified')
        
        # Get template and renderoutput_path
        template_chosen = random.choice(self.templates)
        template = Template(template_chosen)
        label_description = None
        try:
            label_description = template.render(product_info=product_info, 
                                                get_category=self.get_category, 
                                                get_one_hundred_grams=self.get_one_hundred_grams, 
                                                get_serving_size_label=self.get_serving_size_label,
                                                #made_in=self.made_in,
                                                ingredient_label=self.ingredient_label,
                                                nutritional_label=self.nutritional_label,
                                                _nutrient_separator=self._nutrient_separator,
                                                get_nutrient_name=self.get_nutrient_name)
        except Exception as e:
            traceback.print_exc()
            print(e)
            pass
        return label_description
    
    def get_nutrient_name(self, data):
        
        name = data['nutrient']['name']
        name = name.replace('-', ' ').replace('total:', '').replace('Total', '').replace('total', '').replace('by', '').replace('difference', '')
        #if random.random() > .5 and ',' in name:
        if ',' in name:
            if random.random() > .5:
                if random.random() > .5 :
                    name = name.split(',')[0].strip()
                else:
                    name = name.replace(',', ' ').strip()
            else:
                # split by comma and reverse the items
                items = name.split(',')
                items.reverse()
                name = ' '.join(items).strip()
        else:
            name = name.strip()
                                
        return name    
        
    def nutritional_label(self):
        return f'{random.choice(["Nutri Facts:", "Nutrition Facts:", "Nutrition facts:","Nutritional Information:", "NUTRITIONAL INFORMATION:", "NUTRITIONAL INFORMATION", "Nutritional Information", ""])}'

    def _nutrient_separator(self):
        return random.choice(["-", "|", " | ", ' ', "_", "*"])
    
    def ingredient_label(self):
        return f'{random.choice(["Ingredients:", "INGREDIENTS:", "INGREDIENTS", "Ingredients", ""])}'

    def get_serving_size_label(self, serving_size, unit="g", wrap_with_tag=True, ):
        label_1 = f"Service per package {random.randint(2, 8)} | Serving size {serving_size}{unit}"
        label_2 = f"Serving size {serving_size}{unit}"
        label_3 = f"Serving size {serving_size}{unit}"
        label_4 = f"{random.randint(2, 10)} Sachets per serving"
        label_5 = f"{random.randint(2, 10)} Sachets per serving, Serving size {serving_size}{unit}" 
        
        if wrap_with_tag:
            return f' <SERVING_SIZE> {random.choice([label_1, label_2, label_3, label_4, label_5])} </SERVING_SIZE> '
 
        return f'{random.choice([label_1, label_2, label_3, label_4, label_5])}'
    
    
    def get_category(self,category_id):
        query_result = self.nutrients.query(f'id == {category_id}')
        if not query_result.empty:
            return query_result.iloc[0]['Category'].upper().rstrip()
        return random.choice(self.nutrients['Category'].unique()).upper().rstrip()

    def get_one_hundred_grams(self, serving_size=25, serving=1):
        if serving_size == 0 or serving == None:
            return 0   
        plot = 100 / serving_size
        return "{:.1f}".format( plot * serving)
    
    def __ajust_weights(self, data):
        label_frequencies = {
            "O": "0",
            "B-ACIDITYREGULATORS": 1,
            "I-ACIDITYREGULATORS": 2,
            "B-ACIDS": 3,
            "I-ACIDS": 4,
            "B-ADDITIVES": 5,
            "I-ADDITIVES": 6,
            "B-ALCOHOLS": 7,
            "I-ALCOHOLS": 8,
            "B-AMINOACIDS": 9,
            "I-AMINOACIDS": 10,
            "B-CARBOHYDRATES": 11,
            "I-CARBOHYDRATES": 12,
            "B-CAROTENOIDS": 13,
            "I-CAROTENOIDS": 14,
            "B-COLORANTS": 15,
            "I-COLORANTS": 16,
            "B-FIBER": 17,
            "I-FIBER": 18,
            "B-FLAVONOIDS": 19,
            "I-FLAVONOIDS": 20,
            "B-FLAVORINGS": 21,
            "I-FLAVORINGS": 22,
            "B-LIPIDS": 23,
            "I-LIPIDS": 24,
            "B-MACRONUTRIENTS": 25,
            "I-MACRONUTRIENTS": 26,
            "B-MINERALS": 27,
            "I-MINERALS": 28,
            "B-ORGANICCOMPOUNDS": 29,
            "I-ORGANICCOMPOUNDS": 30,
            "B-PHYTOCHEMICALS": 31,
            "I-PHYTOCHEMICALS": 32,
            "B-PRESERVATIVES": 33,
            "I-PRESERVATIVES": 34,
            "B-PROTEINS": 35,
            "I-PROTEINS": 36,
            "B-PROXIMATES": 37,
            "I-PROXIMATES": 38,
            "B-STABILIZERS": 39,
            "I-STABILIZERS": 40,
            "B-STIMULANTS": 41,
            "I-STIMULANTS": 42,
            "B-SUGARS": 43,
            "I-SUGARS": 44,
            "B-VITAMINS": 45,
            "I-VITAMINS": 46,
            "B-WATER": 47,
            "I-WATER": 48
            }
        for weight in label_frequencies:
            label_frequencies[weight] = 0
        for record in data:
            weights = record['weights']
            for key in weights.keys():
                label_frequencies[key] += int(weights[key])
                
        # Calculating inverse frequencies as weights
        label_weights = {label: 1.0 / freq if freq != 0 else 0.0 for label, freq in label_frequencies.items()}
        # Optionally normalize these weights
        total = sum(label_weights.values())
        label_weights = {label: weight / total for label, weight in label_weights.items()}
        return label_weights
    
    def save_state(self, states):
        if len(states) % 100 == 0:
            print(f"Currently at {len(states)}")  
        tokens = []
        index = 0
        try:
            for row in tqdm.tqdm( states, total=len(states), desc='Tokenizing...'):
                iob = row['iob']
                weights = row['weights']
                tokenized_sentence, labels = self.tokenize_and_preserve_labels(iob)
                
                
                
                # step 2: add special tokens (and corresponding labels)
                tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"] # add special tokens
                labels.insert(0, "O") # add outside label for [CLS] token
                labels.insert(-1, "O") # add outside label for [SEP] token
                
                if (len(tokenized_sentence) > self.max_len):
                    # truncate
                    tokenized_sentence = tokenized_sentence[:self.max_len]
                    labels = labels[:self.max_len]
                else:
                    # pad
                    tokenized_sentence = tokenized_sentence + ['[PAD]'for _ in range(self.max_len - len(tokenized_sentence))]
                    labels = labels + ["O" for _ in range(self.max_len - len(labels))]

                    # step 4: obtain the attention mask
                    attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]
                    
                    # step 5: convert tokens to input ids
                    ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)

                    #label_ids = [self.label2id[label.upper()] for label in labels]
                    label_ids = [self.label2id[label.upper()] if isinstance(self.label2id[label.upper()], int) else 0 for label in labels]

                    tokens.append({
                        'input_ids': torch.tensor(ids, dtype=torch.long),
                        'attention_mask': torch.tensor(attn_mask, dtype=torch.long),
                        'labels': torch.tensor(label_ids, dtype=torch.long),
                        'weights': weights
                    })
                    index += 1

        except Exception as e:
            print(f"An error occurred: {e}")
            raise e  # re-raise the exception to stop execution

        
        dataset_balanced_weights = self.__ajust_weights(tokens) 
        # save dataset_balanced_weights to json file
        with open(os.path.join(self.output_dir, f"weights.json"), 'w') as f:
            json.dump(dataset_balanced_weights, f)
            
        with open(os.path.join(self.output_dir, f"labels2ids.json"), 'w') as f:
            json.dump(self.label2id, f )
            
        pickle_file = os.path.join(self.output_dir, f"train_{self.index_file_current_count}.pkl")
        
        with open(pickle_file, 'wb') as f:
            self.index_file_current_count = self.index_file_current_count + 1
            try:
                print('Write Pickle file out.')
                pickle.dump(tokens, f)
            except Exception as e:
                pass

        states.clear()

        
    def _generate_iob(self, raw_string):
        if '<>' in raw_string:
            return None, None
        weights = {
            "O": "0",
            "B-ACIDITYREGULATORS": 1,
            "I-ACIDITYREGULATORS": 2,
            "B-ACIDS": 3,
            "I-ACIDS": 4,
            "B-ADDITIVES": 5,
            "I-ADDITIVES": 6,
            "B-ALCOHOLS": 7,
            "I-ALCOHOLS": 8,
            "B-AMINOACIDS": 9,
            "I-AMINOACIDS": 10,
            "B-CARBOHYDRATES": 11,
            "I-CARBOHYDRATES": 12,
            "B-CAROTENOIDS": 13,
            "I-CAROTENOIDS": 14,
            "B-COLORANTS": 15,
            "I-COLORANTS": 16,
            "B-FIBER": 17,
            "I-FIBER": 18,
            "B-FLAVONOIDS": 19,
            "I-FLAVONOIDS": 20,
            "B-FLAVORINGS": 21,
            "I-FLAVORINGS": 22,
            "B-LIPIDS": 23,
            "I-LIPIDS": 24,
            "B-MACRONUTRIENTS": 25,
            "I-MACRONUTRIENTS": 26,
            "B-MINERALS": 27,
            "I-MINERALS": 28,
            "B-ORGANICCOMPOUNDS": 29,
            "I-ORGANICCOMPOUNDS": 30,
            "B-PHYTOCHEMICALS": 31,
            "I-PHYTOCHEMICALS": 32,
            "B-PRESERVATIVES": 33,
            "I-PRESERVATIVES": 34,
            "B-PROTEINS": 35,
            "I-PROTEINS": 36,
            "B-PROXIMATES": 37,
            "I-PROXIMATES": 38,
            "B-STABILIZERS": 39,
            "I-STABILIZERS": 40,
            "B-STIMULANTS": 41,
            "I-STIMULANTS": 42,
            "B-SUGARS": 43,
            "I-SUGARS": 44,
            "B-VITAMINS": 45,
            "I-VITAMINS": 46,
            "B-WATER": 47,
            "I-WATER": 48
            }
        for weight in weights:
            weights[weight] = 0
            
        
        words = raw_string.split()
        iob = []
        current_tag = []
        for word in words:
            # trim word
            word = word.strip()
            if  word.startswith("</") and word.endswith(">"):
                current_tag = []
            elif word.startswith("<") and word.endswith(">"):
                current_tag.append(word[1:-1])
            else:
                if len(current_tag) == 0:
                    iob.append((word, "O"))
                else:
                    if len(current_tag) == 1:
                        label_ = f"B-{current_tag[0]}".upper()
                        if label_ not in weights:
                            weights[label_] = 1
                        else:
                            weights[label_] += 1
                        # check if current_tag[0] is in self.label2id  
                       
                        # if label_ not in self.label2id:
                        #     label_ = "O"
                        iob.append((word, label_))
                        #if current_tag[0] == label:
                            
                        current_tag.append(current_tag[0])
                    else:
                     
                        label_ = f"I-{current_tag[0]}".upper()
                        if label_ not in weights:
                            weights[label_] = 1
                        else:
                            weights[label_] += 1
                        iob.append((word, label_))

        return iob, weights
    
    
    def tokenize_and_preserve_labels(self, state):
        tokenized_sentence = []
        labels = []

        #sentence = sentence.strip()

        for word, label in state:

            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = self.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)

            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)

            # Add the same label to the new list of labels `n_subwords` times
            labels.extend([label] * n_subwords)

        return tokenized_sentence, labels
    
    def generate_paralell(self):
        num_threads =1  # Adjust the number of threads

        self.process_in_threads(num_threads)

    
    
    
def create_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer
