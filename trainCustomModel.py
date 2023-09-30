import pandas as pd
import numpy as np
import spacy
from sklearn.model_selection import train_test_split
from spacy.training.example import Example
from tqdm import tqdm
import time

ITERATIONS = 100
DROP = 0.5
TRAIN_TEST_SPLIT = 0.3
RANDOM_STATE = 42
WINE_DATA_FILE_PATH = './data/winemag-data-130k-v2 copy.csv'
FOOD_DATA_FILE_PATH = './data/food_keywords.csv'
DATA_SELECTION = 1000

def tastesAndFood(descriptions_list):
    # Load spaCy's English language model
    nlp = spacy.load("en_core_web_lg")

    csv_path = FOOD_DATA_FILE_PATH
    # Read the CSV file into a DataFrame
    food_keywords_set = set(pd.read_csv(csv_path))
    # Initialize a list to store relevant information (nouns and adjectives)
    foodPairing = set()
    tastesLike = set()

    # Iterate through sentences and add matches to the set
    for tastesLikeWord in descriptions_list:
        # Extract food-related phrases from each sentence
        for eachFoodPairing in food_keywords_set:
            if eachFoodPairing in tastesLikeWord:
                foodPairing.add(eachFoodPairing.lower())

        # Process the sentence with spaCy
        doc = nlp(tastesLikeWord)

        # Initialize a set to store doubled-barreled words
        doubled_barreled_words = set()
        word_pairs = set()

        # Iterate through the tokens and check for the desired pattern
        for i in range(len(doc) - 1):
            previous_token = doc[i - 1]
            current_token = doc[i]
            next_token = doc[i + 1]

            # Check if the current token is a word, the next token starts with a capital letter,
            # and neither the current nor the next token contain hyphens
            if (
                current_token.is_alpha
                and next_token.is_alpha
                and next_token.text[0].isupper()
                and "-" not in current_token.text
                and "-" not in next_token.text
            ):
                word_pairs.add((current_token.text + " " + next_token.text))

            # Check for doubled-barreled words
            if "-" in current_token.text:
                doubled_barreled_words.add(previous_token.text + current_token.text + next_token.text)

        for nextTokenCapital in word_pairs:
            tastesLikeWord = tastesLikeWord.replace(nextTokenCapital, "")

        # Remove doubled-barreled words from the sentence
        for doubled_word in doubled_barreled_words:
            tastesLikeWord = tastesLikeWord.replace(doubled_word, "")

        # Process the modified sentence with spaCy
        doc = nlp(tastesLikeWord)

        # Iterate through the tokens and identify relevant information
        for token in doc:
            # Check if the token is a noun (NOUN) or an adjective (ADJ)
            if token.pos_ in ("NOUN", "ADJ"):
                tastesLike.add(token.text.lower())

    # Remove food from tastesLike
    tastesLike = tastesLike.difference(foodPairing)

    # Remove single characters from tastesLike
    tastesLike = dropSingleCharacters(tastesLike)

    return tastesLike, foodPairing

def dropSingleCharacters(inputList):
    removeSingleCharList = [item for item in inputList if len(item) > 1]
    outputList = {element.lower() for element in removeSingleCharList}
    return outputList

def readCSV():
    # Read CSV from filepath
    df = pd.read_csv(WINE_DATA_FILE_PATH)

    descriptions_list = dropSingleCharacters(df[['description']].stack().dropna())
    locations_list = dropSingleCharacters(set(df[['country', 'province', 'region_1', 'region_2']].stack().dropna()))
    designation_list = dropSingleCharacters(set(df[['designation']].stack().dropna()))
    variety_list = dropSingleCharacters(set(df[['variety']].stack().dropna()))
    winery_list = dropSingleCharacters(set(df[['winery']].stack().dropna()))
    redOrWhite_list = {'red', 'white'}
    dryOrSweet_list = {'dry', 'sweet'}
    bodied_list = {'light-bodied', 'medium-bodied', 'full-bodied', 'light bodied', 'medium bodied', 'full bodied'}
    crispOrSmooth_list = {'crisp', 'smooth'}
    tastesLike_list, foodPairing_list = tastesAndFood(descriptions_list)
    
    # Create dictionary mappings for labels
    labelsList = {
        'REGION': locations_list,
        'DESIGNATION': designation_list,
        'VARIETY': variety_list,
        'WINERY': winery_list,
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
    #print(entities_dict)
    return description, entities_dict

def find_substring_positions(main_string, substring, label):
    start_positions = [start_pos for start_pos in range(len(main_string)) if main_string.startswith(substring, start_pos)]
    end_positions = [start_pos + len(substring) for start_pos in start_positions]
    return list(zip(start_positions, end_positions, [label] * len(start_positions)))

def createInputData(descriptionsList, labelsList):
    result = []
    # Loop through the 'descriptions' list
    for description in descriptionsList[:DATA_SELECTION]:
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
    for _ in range(ITERATIONS):  # Adjust the number of iterations
        losses = {}
        with tqdm(total=len(train_data), desc="Training") as pbar:
            for text, annotations in train_data:
                example = Example.from_dict(nlp.make_doc(text), annotations)
                nlp.update([example], drop=DROP, losses=losses)
                pbar.update(1)  # Update the progress bar
        print(losses)

    print("Model Trained")
    nlp.to_disk("wine_ner_model")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total Execution Time: {total_time} seconds")

model()