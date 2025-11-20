import pandas as pd

figsize = {
    "Swear_fluency": (22, 8),
    "Dados_Italian_2": (22, 8),
    "German_data": (22, 8),
    "Parkinson_paper": (22, 8),
    "CPN120": (22, 8),
    "UFABC_PLT_combined": (22, 8),
}

ylabels = {
    "acc_magnitude": "Acceleration",
    "vel_magnitude": "Velocity",
    "distance_centroid_static": "Distance to Centroid",
    "distance_next": "Distance to Next",
    "entropy": "Entropy",
}

categories = {
    "UFABC_PLT_combined": {
        "ABSTRACT_CONCEPT": "Abstract Concept",
        "ABSTRACT_VERB": "Abstract Verb",
        "CONCRETE_CONCEPT": "Concrete Concept",
        "CONCRETE_VERB": "Concrete Verb",
        "HAND_VERB": "Hand Verb",
    },
    "Swear_fluency": {
        "ANIMAL": "Animal",
        "A_LETTER": "Letter A",
        "F_LETTER": "Letter F",
        "S_LETTER": "Letter S",
        "SWEAR_WORDS": "Swear W.",
    },
    "CPN120": {
        "ABSTRACT": "Abstract",
        "CONCRETE": "Concrete",
    },
    "Dados_Italian_2": {
        "bird": "Bird",
        "bodypart": "Body Part",
        "building": "Building",
        "clothing": "Clothing",
        "fruit": "Fruit",
        "furniture": "Furniture",
        "implement": "Implement",
        "mammal": "Mammal",
        "vegetable": "Vegetable",
        "vehicle": "Vehicle",
    },
    "German_data": {
        "bird": "Bird",
        "bodypart": "Body Part",
        "building": "Building",
        "clothing": "Clothing",
        "fruit": "Fruit",
        "furniture": "Furniture",
        "implement": "Implement",
        "mammal": "Mammal",
        "vegetable": "Vegetable",
        "vehicle": "Vehicle",
    },
    "Parkinson_paper": {
        "CN": "HC",
        "DF": "bvFTD",
        "PD": "PD",
    },
}


def stars(p):
    if pd.isna(p):
        return "ns"
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"
