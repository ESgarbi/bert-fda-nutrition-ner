
import random
from utility import generate_nutrient_value

class FoodLabelFakerFattyAcids:
    def __init__(self):
        self.components_fatty_acids = {
        "C4": ("%T", "Butyric Acid (C4)"),
        "C6": ("%T", "Caproic Acid (C6)"),
        "C8": ("%T", "Caprylic Acid (C8)"),
        "C10": ("%T", "Capric Acid (C10)"),
        "C11": ("%T", "Undecylic Acid (C11)"),
        "C12": ("%T", "Lauric Acid (C12)"),
        "C13": ("%T", "Tridecylic Acid (C13)"),
        "C14": ("%T", "Myristic Acid (C14)"),
        "C15": ("%T", "Pentadecylic Acid (C15)"),
        "C16": ("%T", "Palmitic Acid (C16)"),
        "C17": ("%T", "Margaric Acid (C17)"),
        "C18": ("%T", "Stearic Acid (C18)"),
        "C19": ("%T", "Nonadecylic Acid (C19)"),
        "C20": ("%T", "Arachidic Acid (C20)"),
        "C21": ("%T", "Heneicosylic Acid (C21)"),
        "C22": ("%T", "Behenic Acid (C22)"),
        "C23": ("%T", "Tricosylic Acid (C23)"),
        "C24": ("%T", "Lignoceric Acid (C24)"),
        "Total saturated fatty acids": ("%T", "Total Saturated Fatty Acids"),
        "C10:1": ("%T", "Caproleic Acid (C10:1)"),
        "C14:1": ("%T", "Myristoleic Acid (C14:1)"),
        "C15:1": ("%T", "Pentadecenoic Acid (C15:1)"),
        "C16:1": ("%T", "Palmitoleic Acid (C16:1)"),
        "C17:1": ("%T", "Heptadecenoic Acid (C17:1)"),
        "C18:1": ("%T", "Oleic Acid (C18:1)"),
        "C18:1w6": ("%T", "Linoleic Acid (C18:1w6)"),
        "C20:1": ("%T", "Eicosenoic Acid (C20:1)"),
        "C20:1w11": ("%T", "Gondoic Acid (C20:1w11)"),
        "C22:1": ("%T", "Erucic Acid (C22:1)"),
        "C24:1": ("%T", "Nervonic Acid (C24:1)"),
        "Total monounsaturated fatty acids": ("%T", "Total Monounsaturated Fatty Acids"),
        "C18:2w6": ("%T", "Linoleic Acid (C18:2w6)"),
        "C18:3w3": ("%T", "Alpha-Linolenic Acid (ALA, C18:3w3)"),
        "C18:3w6": ("%T", "Gamma-Linolenic Acid (GLA, C18:3w6)"),
        "C18:4w3": ("%T", "Stearidonic Acid (SDA, C18:4w3)"),
        "C20:2w6": ("%T", "Eicosadienoic Acid (C20:2w6)"),
        "C20:3w3": ("%T", "Eicosatrienoic Acid (C20:3w3)"),
        "C20:3w6": ("%T", "Dihomo-gamma-linolenic Acid (DGLA, C20:3w6)"),
        "C20:4w3": ("%T", "Eicosatetraenoic Acid (C20:4w3)"),
        "C20:4w6": ("%T", "Arachidonic Acid (AA, C20:4w6)"),
        "C20:5w3": ("%T", "Eicosapentaenoic Acid (EPA, C20:5w3)"),
        "C22:2w6": ("%T", "Docosadienoic Acid (C22:2w6)"),
        "C22:4w6": ("%T", "Docosatetraenoic Acid (C22:4w6)"),
        "C22:5w3": ("%T", "Docosapentaenoic Acid (DPA, C22:5w3)"),
        "C22:6w3": ("%T", "Docosahexaenoic Acid (DHA, C22:6w3)"),
        "Total polyunsaturated fatty acids": ("%T", "Total Polyunsaturated Fatty Acids"),
        "Total long chain omega 3 fatty acids": ("mg", "Total Long Chain Omega 3 Fatty Acids"),
        "Undifferentiated fatty acids": ("%T", "Undifferentiated Fatty Acids"),
        "Total trans fatty acids": ("%T", "Total Trans Fatty Acids")
    }

    

    
    def get_fatty_acid_food_labels(self):
        # Generate random values for each component

        return list(self.components_fatty_acids.items())