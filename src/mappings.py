import pandas as pd

figsize = {
    "swear-fluency": (22, 8),
    "italian": (22, 8),
    "german": (22, 8),
    "parkinson": (22, 8),
    "CPN120": (22, 8),
    "UFABC_PLT_combined": (22, 8),
}

ylabels = {
    "acc": "Acceleration",
    "vel": "Velocity",
    "d_centroid": "Distance to Centroid",
    "d_next": "Distance to Next",
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
    "swear-fluency": {
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
    "italian": {
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
    "german": {
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
    "parkinson": {
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
