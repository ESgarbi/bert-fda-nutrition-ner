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

class ProductLabelDataGenerator:
    def __init__(self, nutrients_table_path,  json_data_source, augmented_context=[], is_validation_set=False, total_lines=10000,output_path_name=None):
        self.materialized_data = {}
        self.counter = 1
        print('got inside ProductLabelDataGenerator')
        self.augmented_context = augmented_context
        print(nutrients_table_path)
        self.nutrients = pd.read_csv(nutrients_table_path)
        print('Rad Nutrients')
        self.nutrients = self.nutrients.set_index('id')  
        self.state_file = os.path.join(output_path_name if output_path_name else 'default/path', 'state.parquet')
        self.master_templates = '''{{ product_info.description }}
                    {{ ingredient_label() }} <INGREDIENTS> {{ product_info.ingredients }} </INGREDIENTS>
                {{ nutritional_label() }}

                {% for nutrient in product_info['foodNutrients'] %}

                <{{ get_category(nutrient.nutrient.id) }}> {{ get_nutrient_name(nutrient.nutrient.name) }}
                {{ nutrient.amount }}
                {{ nutrient.nutrient.unitName }}
                </{{ get_category(nutrient.nutrient.id) }}>
                {{ get_one_hundred_grams(product_info.servingSize, nutrient.amount) }}{{ nutrient.nutrient.unitName }}
                {% endfor %}
                <PACKAGE_WEIGHT> {{ product_info['packageWeight'] }} </PACKAGE_WEIGHT>
                '''
        self.data = self.generate_raw_base(json_data_source, is_validation_set, total_lines, output_path_name=output_path_name)
        self.templates = ['''{{ product_info.description }}
                {{ ingredient_label() }} <INGREDIENTS> {{ product_info.ingredients }} </INGREDIENTS>
                {{ nutritional_label() }}
                {% for nutrient in product_info['foodNutrients'] %}
                <{{ get_category(nutrient.nutrient.id) }}> {{ get_nutrient_name(nutrient.nutrient.name) }} {{ nutrient.amount }} {{ nutrient.nutrient.unitName }} </{{ get_category(nutrient.nutrient.id) }}>
                {% endfor %}
                <PACKAGE_WEIGHT> {{ product_info['packageWeight'] }} </PACKAGE_WEIGHT>
                ''',
                '''{{ product_info.description }}
                {{ get_one_hundred_grams(product_info.servingSize, nutrient.amount) }}{{ nutrient.nutrient.unitName }}
                {{ ingredient_label() }} <INGREDIENTS> {{ product_info.ingredients }} </INGREDIENTS>
                {{ nutritional_label() }}
                {% for nutrient in product_info['foodNutrients'] %}
                <{{ get_category(nutrient.nutrient.id) }}> {{ get_nutrient_name(nutrient.nutrient.name) }} {{ nutrient.amount }} </{{ get_category(nutrient.nutrient.id) }}>
                {% endfor %}
                <PACKAGE_WEIGHT> {{ product_info['packageWeight'] }} </PACKAGE_WEIGHT>
                '''
                ]
   
    def _misspell_word(self, word):
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        index = random.randint(0, len(word) - 1)
        new_letter = random.choice(alphabet)
        return word[:index] + new_letter + word[index + 1:]   

    def _add_random_letter(self, word):
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        index = random.randint(0, len(word))
        new_letter = random.choice(alphabet)
        return word[:index] + new_letter + word[index:]
 
    # this method will randomnly delete letters from a word
    def _delete(self, word):
        if random.random() > .7:
            return word
        else:
            return word[:random.randint(0, len(word))] + word[random.randint(0, len(word)):]   
        
    def get_nutrient_name(self, name):
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
        
        if random.random() > .5:
           # choose randomly between _misspell_word, _delete and _add_random_letter or all of them and return the result
            if random.random() > .5:
                name = self._misspell_word(name)
            if random.random() > .5:
                name = self._delete(name)
            if random.random() > .5:
                name = self._add_random_letter(name)
                    
        return name    

            
    def _nutrient_separator(self):
        return random.choice(["-", "|", " | ", ' ', "_", "*"])
    
    
    def _generate_iob(self, raw_string):
        
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
                        iob.append((word, f"B-{current_tag[0]}"))
                        current_tag.append(current_tag[0])
                    else:
                        iob.append((word, f"I-{current_tag[0]}"))

        return iob
    
  
    def generate_raw_base(self, json_data_source, is_validation_set, total_lines=50000, estimated_line_length=100, output_path_name=None):
        lines = []
        state_file_df = None
        if output_path_name != None and os.path.exists(os.path.join(output_path_name, 'state.parquet')):
            state_path = os.path.join(output_path_name, 'state.parquet')
            saved_state = pd.read_parquet(state_path)
            total_saved_count = len(saved_state)
            saved_state = saved_state[saved_state['is_trained'] == False]
            print(f'Found {total_saved_count} records in state file, {len(saved_state)} are not trained yet')
            return saved_state
        else:    
            lines = []
            
            with open(json_data_source, 'r') as f:
                count = 0
                for line in tqdm.tqdm_notebook(f, total=total_lines, desc='Generating raw data'):
                    try:
                        count = count + 1
                        
                        if count > total_lines:
                            break
                        # show progress eery 10K files
                        if count % 10000 == 0:
                            print(f'{count} of {total_lines} processed')
                        line = line.rstrip(" ").rstrip(", ").rstrip("'").rstrip(' ').rstrip(",\n'")
                        
                        try:
                            json_line = json.loads(line)
                            data_label = ''
                            # data_label = self._sequence_augmentation_1(json_line)
                            # lines.append(self.augument(data_label))
                            if random.random() > .5:
                                data_label = self._sequence_augmentation_1(json_line)
                            else:
                                data_label = self._sequence_augmentation_0(json_line)
                            lines.append(self.augument(data_label))

                        except Exception as e:
                            pass
                        # Check if the line is valid and add to list
                       
                        
                    except: 
                        pass
                lines = [(line, False) for line in lines]
                
                state_file_df = pd.DataFrame(lines, columns=['product_label_raw','is_trained'])
                if not is_validation_set:
                    # Only save state if it is training data, not validation.
                    if not os.path.exists(output_path_name):
                        os.makedirs(output_path_name)
                        
                    state_file_df.to_parquet(os.path.join(output_path_name, 'state.parquet'))
        return state_file_df
    
    # def generate_raw_base(self, json_data_source, total_lines=50000):
    #     lines = []
        
    #     current =  0
    #     with open(json_data_source, 'r') as f:
    #         next(f)
    #         pbar = tqdm.tqdm_notebook(total=total_lines)
    #         while True:
    #             current = current + 1
    #             if current > total_lines:
    #                 df = pd.DataFrame(lines, columns=['product_label_raw'])
    #                 return df
    #             line = f.readline()
    #             line = line.rstrip(" ").rstrip(", ").rstrip("'").rstrip(' ').rstrip(",\n'")
    #             if current > total_lines:
    #                 break
    #             lines.append(line)
    #             pbar.update(1)
                
    
    def nutritional_label(self):
        return f'{random.choice(["Nutri Facts:", "Nutrition Facts:", "Nutrition facts:","Nutritional Information:", "NUTRITIONAL INFORMATION:", "NUTRITIONAL INFORMATION", "Nutritional Information", ""])}'

    
    def ingredient_label(self):
        return f'{random.choice(["Ingredients:", "INGREDIENTS:", "INGREDIENTS", "Ingredients", ""])}'

    
    def get_one_hundred_grams(self, serving_size=25, serving=1):
        if serving_size == 0 or serving == None:
            return 0   
        plot = 100 / serving_size
        return "{:.1f}".format( plot * serving)
   
    def get_category(self,category_id):
        
        if int(category_id) in self.nutrients.index:
            # do a random 30 percent chance of returning the category
            
            return self.nutrients.loc[category_id]['Category'].upper().rstrip()
        return random.choice(self.nutrients['Category'].unique()).upper().rstrip()

    def get_ingredient_name(self,category_id):
       
        if int(category_id) in nutrients.index:
            return nutrients.loc[category_id]['name'].upper()
        return "OTHER"
    def made_in(self, country):
        label_1 = f"Made in <COUNTRY_ORIGIN> {country} </COUNTRY_ORIGIN> "
        label_2 = f"Made in <COUNTRY_ORIGIN> {country} </COUNTRY_ORIGIN> , Distributed by "
        label_3 = f"Product of <COUNTRY_ORIGIN> {country} </COUNTRY_ORIGIN> "
        label_4 = f"Product of <COUNTRY_ORIGIN> {country} </COUNTRY_ORIGIN> , Distributed by "
        label_5 = f"Produced in <COUNTRY_ORIGIN> {country} </COUNTRY_ORIGIN> "
        label_6 = f"Sourced from <COUNTRY_ORIGIN> {country} </COUNTRY_ORIGIN> "
        
        return f'{random.choice([label_1, label_2, label_3, label_4, label_5, label_6])}'

    def get_serving_size_label(self, serving_size, unit="g", wrap_with_tag=True, ):
        label_1 = f"Service per package {random.randint(2, 8)} | Serving size {serving_size}{unit}"
        label_2 = f"Serving size {serving_size}{unit}"
        label_3 = f"Serving size {serving_size}{unit}"
        label_4 = f"{random.randint(2, 10)} Sachets per serving"
        label_5 = f"{random.randint(2, 10)} Sachets per serving, Serving size {serving_size}{unit}" 
        
        if wrap_with_tag:
            return f' <SERVING_SIZE> {random.choice([label_1, label_2, label_3, label_4, label_5])} </SERVING_SIZE> '
        
        return f'{random.choice([label_1, label_2, label_3, label_4, label_5])}'
    
    def _sequence_augmentation_1(self, data):
        """
        Formats the food data into an ASCII table representation.
        """
        from prettytable import PrettyTable

        # Create a table for basic information
        basic_info_table = PrettyTable()
        basic_info_table.field_names = ["Attribute", "Value"]
        basic_info_table.align["Attribute"] = "l"
        basic_info_table.align["Value"] = "l"

        basic_info_attributes = [
            ("Food Class", data['foodClass']),
            ("Description", data['description']),
            ("Brand Owner", data['brandOwner']),
            ("GTIN/UPC", data['gtinUpc']),
            ("Data Source", data['dataSource']),
            ("Ingredients", ' <INGREDIENTS> ' + data['ingredients'] + ' </INGREDIENTS> '),
            # ("Serving Size", self.get_serving_size_label(data['servingSize'], data['servingSizeUnit']))
        ]

        for attr, value in basic_info_attributes:
            basic_info_table.add_row([attr, value])

        # Create a table for nutrients
        nutrients_table = PrettyTable()
        nutrients_table.field_names = ["Nutrient", "Amount", "Unit"]
        nutrients_table.align = "l"

        for nutrient in data['foodNutrients']:
            category = self.get_category(nutrient['id'])
            nutrient_name =  nutrient['nutrient']['name'] 
            amount = nutrient['amount']
            unit = nutrient['nutrient']['unitName']
            nutrients_table.add_row([f" <{category}> " + nutrient_name, amount, unit + f" </{category}> "])
        compiled_label = f"{basic_info_table}\n\nNutrients:\n{nutrients_table}"
        if 'packageWeight' in data:
            compiled_label = compiled_label +  f'''\n <PACKAGE_WEIGHT> {data['packageWeight']} </PACKAGE_WEIGHT> '''
        return compiled_label
    
    def _sequence_augmentation_0(self, product_info):
        #product_info = json.loads(self.data.iloc[index]['product_label_raw'])
        
        product_info['description'] = product_info.get('description', 'No description available')
        product_info['brandOwner'] = product_info.get('brandOwner', 'Unknown brand owner')
        product_info['ingredients'] = product_info.get('ingredients', 'Ingredients not available')
        product_info['marketCountry'] = product_info.get('marketCountry', 'Unknown market country')
        product_info['servingSize'] = product_info.get('servingSize', 'Serving size not specified')



        # Get template and renderoutput_path
        template_chosen = random.choice(self.templates)
        template = Template(template_chosen)

        label_description = template.render(product_info=product_info, 
                                            get_category=self.get_category, 
                                            get_one_hundred_grams=self.get_one_hundred_grams, 
                                            get_serving_size_label=self.get_serving_size_label,
                                            made_in=self.made_in,
                                            ingredient_label=self.ingredient_label,
                                            nutritional_label=self.nutritional_label,
                                            _nutrient_separator=self._nutrient_separator,
                                            get_nutrient_name=self.get_nutrient_name)

        return label_description
    # def _sequence_augmentation_0(self, product_info):
    #     #product_info = json.loads(self.data.iloc[index]['product_label_raw'])
        
    #     product_info['description'] = product_info.get('description', 'No description available')
    #     product_info['brandOwner'] = product_info.get('brandOwner', 'Unknown brand owner')
    #     product_info['ingredients'] = product_info.get('ingredients', 'Ingredients not available')
    #     product_info['marketCountry'] = product_info.get('marketCountry', 'Unknown market country')
    #     product_info['servingSize'] = product_info.get('servingSize', 'Serving size not specified')

    #     # # Formatting the label nutrients information
    #     # label_nutrients = product_info.get('foodNutrients', {})
    #     # product_info['nutrients_info'] = "\n".join([f" <{self.get_category(value['nutrient']['id'])}> {value['nutrient']['name']} <{self.get_category(value['nutrient']['id'])}> : <quantity> {value['amount']}{value['nutrient'].get('unitName', '')} </quantity>"
    #     #                                             for value in label_nutrients])

    #     # Get template and renderoutput_path
       
    #     template = Template(self.master_templates)

    #     label_description = template.render(product_info=product_info, 
    #                                         get_category=self.get_category, 
    #                                         get_one_hundred_grams=self.get_one_hundred_grams, 
    #                                         get_serving_size_label=self.get_serving_size_label,
    #                                         made_in=self.made_in,
    #                                         ingredient_label=self.ingredient_label,
    #                                         nutritional_label=self.nutritional_label,
    #                                         _nutrient_separator=self._nutrient_separator,
    #                                         get_nutrient_name=self.get_nutrient_name)

    #     return label_description
    
    def save(self):
        print(f'Saving {self.state_file}')
        self.data.to_parquet(self.state_file)
        
    def augument(self, compiled_label):
        context = random.choice(self.augmented_context)
        # choose position between 1 and 4
        position = random.randint(1, 4)
        # get sample from augmented_context
        if len(self.augmented_context) > 0:
            # get text context to mix with the label
            text_context = context
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
        
    def get_data_sequence(self, index, is_validation_set=False):
        if index in self.materialized_data:
            return self.materialized_data[index]
      
        row_data = json.loads(self.data.iloc[index]['product_label_raw'])
        
        self.counter = (self.counter + 1) % 2
        compiled_label = getattr(self, f'_sequence_augmentation_{self.counter}')(row_data)
        self.counter = (self.counter + 1) % 2
        # call self._sequence_augmentation_$ dynamicaly
        compiled_label = getattr(self, f'_sequence_augmentation_{self.counter}')(row_data)
        context = random.choice(self.augmented_context)
        # choose position between 1 and 4
        position = random.randint(1, 4)
        # get sample from augmented_context
        if len(self.augmented_context) > 0:
            # get text context to mix with the label
            text_context = context
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
        
    
        iob = self._generate_iob(compiled_label)
        # iob is the actual main target training data
        # compiled_label is just the label in iob format
        # context is the text context that will be mixed with the label in iob format
         
        self.materialized_data[index] = (iob, compiled_label, self._generate_iob(context))
        
        self.data.loc[index, 'is_trained'] = True
        # if not is_validation_set:
        #     # update the state file
            
        #     self.data.to_parquet(self.state_file)
        
        return self.materialized_data[index]#iob, compiled_label, self._generate_iob(new_context)


def is_macos():
    return platform.system() == 'Darwin'
