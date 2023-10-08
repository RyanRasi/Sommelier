import pandas as pd
import numpy as np
import spacy
from sklearn.model_selection import train_test_split
from spacy.training.example import Example
from tqdm import tqdm
import time
import re
from itertools import islice

ITERATIONS = 100
DROP = 0.5
TRAIN_TEST_SPLIT = 0.3
RANDOM_STATE = 42
WINE_DATA_FILE_PATH = './data/winemag-data-130k-v2 copy.csv'
FOOD_DATA_FILE_PATH = './assets/food_keywords.csv'
DESC_DATA_FILE_PATH = './assets/description_keywords.csv'
DATA_SELECTION = 1430

def dataSanitisation(inputList):
    # Remove single characters
    removeSingleCharList = [item for item in inputList if len(item) > 1]
    # Make lists lowercase
    outputList = {element.lower() for element in removeSingleCharList}
    return outputList

def readCSV():
    # Read CSV from filepath
    df = pd.read_csv(WINE_DATA_FILE_PATH)
    descriptions_list = dataSanitisation(set(df[['description']].stack().dropna()))
    redOrWhite_list = {'red', 'white'}
    dryOrSweet_list = {'dry', 'sweet'}
    bodied_list = {'light-bodied', 'medium-bodied', 'full-bodied', 'light bodied', 'medium bodied', 'full bodied'}
    crispOrSmooth_list = {'crisp', 'smooth'}
    tastesLike_list = dataSanitisation(set(pd.read_csv(DESC_DATA_FILE_PATH)))
    foodPairing_list = dataSanitisation(set(pd.read_csv(FOOD_DATA_FILE_PATH)))
    # Create dictionary mappings for labels
    labelsList = {
        'COLOUR': redOrWhite_list,
        'DRYORSWEET': dryOrSweet_list,
        'BODIED': bodied_list,
        'CRISPORSMOOTH': crispOrSmooth_list,
        'FLAVOUR': tastesLike_list,
        'FOOD': foodPairing_list   
    }
    return descriptions_list, labelsList

def remove_overlapping_entities(entities):
    # Sort entities by start position
    sorted_entities = sorted(entities, key=lambda x: x[0])
    # Use a list to store non-overlapping entities
    non_overlapping_entities = []
    # Iterate through the sorted entities
    for current_entity in sorted_entities:
        # If non_overlapping_entities is empty or the current_entity does not overlap with the last entity in the list
        if not non_overlapping_entities or current_entity[0] >= non_overlapping_entities[-1][1]:
            non_overlapping_entities.append(current_entity)
        else:
            # If the current_entity overlaps, choose the one with the larger span
            if current_entity[1] > non_overlapping_entities[-1][1]:
                non_overlapping_entities[-1] = current_entity
    return non_overlapping_entities

def label_entities(description, Labelslist):
    entities = []

    # Loop through each label and its associated list
    for label, substring_list in Labelslist.items():
        # Loop through each substring in the list
        for substring in substring_list:
            # Find substring positions and add them to the entities list
            positions = find_substring_positions(description, substring, label)
            entities.extend(positions)
        
    # Remove overlapping entitites
    entities = remove_overlapping_entities(entities)
    # Convert entities to the required format
    entities_dict = {"entities": [(start, end, label) for start, end, label in entities]}
    return description, entities_dict

def find_substring_positions(main_string, substring, label):
    # Using word boundaries to find exact matches
    pattern = re.compile(r'\b' + re.escape(substring) + r'\b')
    matches = pattern.finditer(main_string)
    positions = [(match.start(), match.end(), label) for match in matches]
    return positions

def createInputData(descriptionsList, labelsList):
    result = []
    # Loop through the 'descriptions' list
    for description in islice(descriptionsList, DATA_SELECTION):
        result.append(label_entities(description, labelsList))

    return result

def model():
    start_time = time.time()

    descriptionsList, labelsList = readCSV()

    # Split the set into training (70%) and testing (30%) data
    train_data, test_data = train_test_split(list(createInputData(descriptionsList, labelsList)), test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE)

     # Load a blank spaCy model
    nlp = spacy.blank("en")

    # Create the NER component
    ner = nlp.create_pipe("ner")
    # Label entities
    ner.add_label("REGION")
    ner.add_label("DESIGNATION")
    ner.add_label("VARIETY")
    ner.add_label("WINERY")
    ner.add_label("COLOUR")
    ner.add_label("DRYORSWEET")
    ner.add_label("BODIED")
    ner.add_label("CRISPORSMOOTH")
    ner.add_label("FLAVOUR")
    ner.add_label("FOOD")
    
    # Add the NER component to the pipeline
    nlp.add_pipe("ner")

    # Training loop
    optimizer = nlp.begin_training()
    with tqdm(total=ITERATIONS, desc="Training") as pbar:
        for _ in range(ITERATIONS):  # Adjust the number of iterations
            losses = {}
            for text, annotations in train_data:
                example = Example.from_dict(nlp.make_doc(text), annotations)
                nlp.update([example], drop=DROP, losses=losses)
            pbar.update(1)  # Update the progress bar
        print(losses)

    print("Model Trained")
    nlp.to_disk("./assets/wine_ner_model")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total Execution Time: {total_time} seconds")