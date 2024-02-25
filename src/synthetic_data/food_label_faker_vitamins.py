
import random
from utility import generate_nutrient_value

# We'll use the same class FoodLabelFaker but add a method to respect the lower limits
class FoodLabelFakerVitamins:
    def __init__(self):
       self.components = {
        "Retinol (preformed vitamin A)": ("ug", "Vitamin A (Retinol)"),
        "Alpha-carotene": ("ug", "Provitamin A (Alpha-carotene)"),
        "Beta-carotene": ("ug", "Provitamin A (Beta-carotene)"),
        "Cryptoxanthin": ("ug", "Provitamin A (Cryptoxanthin)"),
        "Beta-carotene equivalents (provitamin A)": ("ug", "Provitamin A Equivalents"),
        "Vitamin A retinol equivalents": ("ug", "Vitamin A Retinol Equivalents"),
        "Lutein": ("ug", "Lutein"),
        "Lycopene": ("ug", "Lycopene"),
        "Xanthophyl": ("ug", "Xanthophyll"),
        "Thiamin (B1)": ("mg", "Vitamin B1 (Thiamin)"),
        "Riboflavin (B2)": ("mg", "Vitamin B2 (Riboflavin)"),
        "Niacin (B3)": ("mg", "Vitamin B3 (Niacin)"),
        "Niacin derived from tryptophan": ("mg", "Niacin from Tryptophan"),
        "Niacin derived equivalents": ("mg", "Niacin Equivalents"),
        "Pantothenic acid (B5)": ("mg", "Vitamin B5 (Pantothenic Acid)"),
        "Pyridoxine (B6)": ("mg", "Vitamin B6 (Pyridoxine)"),
        "Biotin (B7)": ("ug", "Vitamin B7 (Biotin)"),
        "Cobalamin (B12)": ("ug", "Vitamin B12 (Cobalamin)"),
        "Folate, natural": ("ug", "Folate (Natural)"),
        "Folic acid": ("ug", "Folic Acid"),
        "Total folates": ("ug", "Total Folates"),
        "Dietary folate equivalents": ("ug", "Dietary Folate Equivalents"),
        "Vitamin C": ("mg", "Vitamin C"),
        "Cholecalciferol (D3)": ("ug", "Vitamin D3 (Cholecalciferol)"),
        "Ergocalciferol (D2)": ("ug", "Vitamin D2 (Ergocalciferol)"),
        "25-hydroxy cholecalciferol (25-OH D3)": ("ug", "25-Hydroxy Vitamin D3"),
        "25-hydroxy ergocalciferol (25-OH D2)": ("ug", "25-Hydroxy Vitamin D2"),
        "Vitamin D3 equivalents": ("ug", "Vitamin D3 Equivalents"),
        "Alpha tocopherol": ("mg", "Vitamin E (Alpha Tocopherol)"),
        "Alpha tocotrienol": ("mg", "Vitamin E (Alpha Tocotrienol)"),
        "Beta tocopherol": ("mg", "Vitamin E (Beta Tocopherol)"),
        "Beta tocotrienol": ("mg", "Vitamin E (Beta Tocotrienol)"),
        "Delta tocopherol": ("mg", "Vitamin E (Delta Tocopherol)"),
        "Delta tocotrienol": ("mg", "Vitamin E (Delta Tocotrienol)"),
        "Gamma tocopherol": ("mg", "Vitamin E (Gamma Tocopherol)"),
        "Gamma tocotrienol": ("mg", "Vitamin E (Gamma Tocotrienol)"),
        "Vitamin E": ("mg", "Vitamin E (Total)"),
    }

    
    
    def generate_vitamin_food_label(self):
        # # Generate random values for each component
        # food_label_data = {
        #     component: generate_nutrient_value(unit) for component, unit in self.components.items()
        # }

        return  list(self.components)
