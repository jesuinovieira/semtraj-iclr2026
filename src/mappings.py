import pandas as pd

figsize = {
    "UFABC_PLT_combined": (6, 8),
    "Swear_fluency": (6, 10),
    "CPN120": (4, 8),
    "Dados_Italian_2": (7, 7),
    "German_data": (8, 8),
    "Parkinson_paper": (4, 6),
}

ylabels = {
    "acc_magnitude": "Acceleration",
    "vel_magnitude": "Velocity",
    "distance_centroid_static": "Distance to centroid",
    "distance_next": "Trajectory",
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
        "SWEAR_WORDS": "Swear Words",
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
