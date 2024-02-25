import random
import faker
from food_label_faker_fatty_acids import FoodLabelFakerFattyAcids
from food_label_faker_other import FoodLabelFakerOther
from food_label_faker_vitamins import FoodLabelFakerVitamins
from food_label_faker_minerals import FoodLabelFakerMinerals


# Creating a 'faker' class to generate food label nutritional data
class FoodLabelFaker:
    def __init__(self):
        # Using the 'faker' library to generate random dates and times
        self.fake = faker.Faker()
        self.fatty_acids = FoodLabelFakerFattyAcids()
        self.other = FoodLabelFakerOther()
        self.vitamins = FoodLabelFakerVitamins()
        self.minerals = FoodLabelFakerMinerals()
        self.weights_strategy = {
            
            'stragegy_1': {
                'fatty_acids': 20,  
                'minerals': 20,     
                'vitamins': 40,     
                'other': 20,        
            },
            'stragegy_2': {
                'fatty_acids': 30,  
                'minerals': 40,     
                'vitamins': 20,     
                'other': 10,        
            },
            'stragegy_3': {
                'fatty_acids': 40,  
                'minerals': 20,     
                'vitamins': 30,     
                'other': 10,        
            },
            'stragegy_4': {
                'fatty_acids': 50,  
                'minerals': 10,     
                'vitamins': 30,     
                'other': 10,        
            },
            'stragegy_5': {
                'fatty_acids': 60,  
                'minerals': 10,     
                'vitamins': 20,     
                'other': 10,        
            },
            'stragegy_6': {
                'fatty_acids': 70,  
                'minerals': 10,     
                'vitamins': 10,     
                'other': 10,        
            },
            'stragegy_7': {
                'fatty_acids': 80,  
                'minerals': 10,     
                'vitamins': 10,     
                'other': 0,        
            },
            'stragegy_8': {
                'fatty_acids': 90,  
                'minerals': 10,     
                'vitamins': 0,     
                'other': 0,        
            },
            'stragegy_9': {
                'fatty_acids': 100,  
                'minerals': 0,     
                'vitamins': 0,     
                'other': 0,        
            }
        }
        
    def generate_label_data(self, total_ingredients):
        
        # get a random strategy
        strategy = random.choice(list(self.weights_strategy.values()))
        # Calculate the number of ingredients per category based on weights
        num_fatty_acids = int((strategy['fatty_acids'] / 100) * total_ingredients)
        num_minerals = int((strategy['minerals'] / 100) * total_ingredients)
        num_vitamins = int((strategy['vitamins'] / 100) * total_ingredients)
        num_others = int((strategy['other'] / 100) * total_ingredients)


        label_fatty_acids = random.sample(self.fatty_acids.get_fatty_acid_food_labels(), num_fatty_acids)
        label_minerals = random.sample(self.minerals.generate_mineral_food_label(), num_minerals)
        label_vitamins = random.sample(self.vitamins.generate_vitamin_food_label(), num_vitamins)
        label_others = random.sample(self.other.generate_other_food_label(), num_others)
        
        label = label_fatty_acids + label_minerals + label_vitamins

        # Shuffle the combined list to randomize the order of ingredients
        random.shuffle(label)

        return {
            'fatty_acids': label_fatty_acids,
            'minerals': label_minerals,
            'vitamins': label_vitamins,
            'other': label,
        }
        


    def generate_nutrient_value(self, unit):
        # Define ranges for nutrient values depending on the unit
        ranges = {
            'kJ': (100, 3000),
            'g': (0, 100),
            'ug': (0, 1000),
            'mg': (0, 500),
            '%T; g': (0, 100),
            '%T; mg': (0, 1000),
            'mg/g N; mg': (0, 100),
        }

        # Get a random value within the range for the given unit
        value_range = ranges.get(unit, (0, 100))
        return round(random.uniform(*value_range), 2)
