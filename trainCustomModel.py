import pandas as pd
import numpy as np
import spacy
from sklearn.model_selection import train_test_split
from spacy.training.example import Example

ITERATIONS = 100
DROP = 0.5
TRAIN_TEST_SPLIT = 0.3
RANDOM_STATE = 42
INPUT_FILE_PATH = './data/winemag-data-130k-v2 copy.csv'

def readCSV():
    # Read CSV from filepath
    df = pd.read_csv(INPUT_FILE_PATH)

    descriptions_list = df[['description']].stack().dropna()
    locations_list = set(df[['country', 'province', 'region_1', 'region_2']].stack().dropna())
    designation_list = set(df[['designation']].stack().dropna())
    variety_list = set(df[['variety']].stack().dropna())
    winery_list = set(df[['winery']].stack().dropna())
    redOrWhite_list = {'Red', 'White'}
    dryOrSweet_list = {'Dry', 'Sweet'}
    bodied_list = {'light-bodied', 'medium-bodied', 'full-bodied', 'light bodied', 'medium bodied', 'full bodied'}
    crispOrSmooth_list = {'Crisp', 'Smooth'}
    tastesLike_list = {}
    foodPairing_list = {}
    # Print the resulting list
    print(locations_list)
    print(designation_list)
    print(variety_list)
    print(winery_list)

test = [
    "This red wine from pairs well with steak."
]
test_country = "Napa Valley"

def find_substring_positions(main_string, substring):
    start_positions = [start_pos for start_pos in range(len(main_string)) if main_string.startswith(substring, start_pos)]
    end_positions = [start_pos + len(substring) for start_pos in start_positions]
    return list(zip(start_positions, end_positions))
# Example
main_string = test[0]
substring = test_country
positions = find_substring_positions(main_string, substring)
print("Positions:", positions)

# Provide annotated training data
input_data = [
    ("This red wine from Napa Valley pairs well with steak.", {"entities": [(19, 30, "REGION"), (51, 56, "FOOD_PAIRING")]}),
    ("The Central Coast offers the best wines for steak.", {"entities": [(4, 17, "REGION"), (44, 49, "FOOD_PAIRING")]}),
    # More annotated examples...
]

# Split the set into training (70%) and testing (30%) data
train_set, test_set = train_test_split(list(input_data), test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE)


def model():
    # Load a blank spaCy model
    nlp = spacy.blank("en")

    # Create the NER component
    ner = nlp.create_pipe("ner")
    # Label entities
    ner.add_label("REGION")
    ner.add_label("FOOD PAIRING")
    # Add the NER component to the pipeline
    nlp.add_pipe("ner")

    # Training loop
    optimizer = nlp.begin_training()
    for _ in range(ITERATIONS):  # Adjust the number of iterations
        losses = {}
        for text, annotations in train_set:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], drop=DROP, losses=losses)
        print(losses)

    # Test custom model
    text = "Wines from Napa Valley and Central Coast are excellent and offer the best pairing for steak."
    doc = nlp(text)
    for ent in doc.ents:
        print(f"Entity: {ent.text}, Label: {ent.label_}")

    nlp.to_disk("wine_ner_model")

#readCSV()