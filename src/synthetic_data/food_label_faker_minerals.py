from utility import generate_nutrient_value
import random


class FoodLabelFakerMinerals:
    def __init__(self):
        self.components_minerals = {
        "Aluminium (Al)": ("ug", "Aluminium"),
        "Antimony (Sb)": ("ug", "Antimony"),
        "Arsenic (As)": ("ug", "Arsenic"),
        "Cadmium (Cd)": ("ug", "Cadmium"),
        "Calcium (Ca)": ("mg", "Calcium"),
        "Chromium (Cr)": ("ug", "Chromium"),
        "Chloride (Cl)": ("mg", "Chloride"),
        "Cobalt (Co)": ("ug", "Cobalt"),
        "Copper (Cu)": ("mg", "Copper"),
        "Fluoride (F)": ("ug", "Fluoride"),
        "Iodine (I)": ("ug", "Iodine"),
        "Iron (Fe)": ("mg", "Iron"),
        "Lead (Pb)": ("ug", "Lead"),
        "Magnesium (Mg)": ("mg", "Magnesium"),
        "Manganese (Mn)": ("mg", "Manganese"),
        "Mercury (Hg)": ("ug", "Mercury"),
        "Molybdenum (Mo)": ("ug", "Molybdenum"),
        "Nickel (Ni)": ("ug", "Nickel"),
        "Phosphorus (P)": ("mg", "Phosphorus"),
        "Potassium (K)": ("mg", "Potassium"),
        "Selenium (Se)": ("ug", "Selenium"),
        "Sodium (Na)": ("mg", "Sodium"),
        "Sulphur (S)": ("mg", "Sulphur"),
        "Tin (Sn)": ("ug", "Tin"),
        "Zinc (Zn)": ("mg", "Zinc")
    }


    def generate_mineral_food_label(self):

        return list(self.components_minerals.items())
    
