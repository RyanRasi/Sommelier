import pandas as pd
import numpy as np
import spacy
from sklearn.model_selection import train_test_split
from spacy.training.example import Example

ITERATIONS = 100
DROP = 0.5
TRAIN_TEST_SPLIT = 0.3
RANDOM_STATE = 42
WINE_DATA_FILE_PATH = './data/winemag-data-130k-v2 copy.csv'
FOOD_DATA_FILE_PATH = './data/food_keywords.csv'

def tastesAndFood(descriptions_list):
    # Load spaCy's English language model
    nlp = spacy.load("en_core_web_lg")

    csv_path = FOOD_DATA_FILE_PATH
    # Read the CSV file into a DataFrame
    food_keywords_set = set(pd.read_csv(csv_path))
    # Initialize a list to store relevant information (nouns and adjectives)
    foodPairing = set()
    tastesLike = set()

    print("Food Keywords Loaded from CSV: ", food_keywords_set)

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

    # Print the relevant information (tastesLike)
    ##print("tastesLike:", tastesLike)

    # Print the extracted food phrases
    ##print("Extracted Food Phrases:", foodPairing)

    return tastesLike, foodPairing

def readCSV():
    # Read CSV from filepath
    df = pd.read_csv(WINE_DATA_FILE_PATH)

    descriptions_list = df[['description']].stack().dropna()
    locations_list = set(df[['country', 'province', 'region_1', 'region_2']].stack().dropna())
    designation_list = set(df[['designation']].stack().dropna())
    variety_list = set(df[['variety']].stack().dropna())
    winery_list = set(df[['winery']].stack().dropna())
    redOrWhite_list = {'Red', 'White'}
    dryOrSweet_list = {'Dry', 'Sweet'}
    bodied_list = {'light-bodied', 'medium-bodied', 'full-bodied', 'light bodied', 'medium bodied', 'full bodied'}
    crispOrSmooth_list = {'Crisp', 'Smooth'}
    tastesLike_list, foodPairing_list = tastesAndFood(descriptions_list)
    # Print the resulting list
    ##print(locations_list)
    ##print(designation_list)
    ##print(variety_list)
    ##print(winery_list)
    ##print(redOrWhite_list)
    ##print(dryOrSweet_list)
    ##print(bodied_list)
    ##print(crispOrSmooth_list)
    ##print(tastesLike_list)
    ##print(foodPairing_list)
    Labelslist = {
        'descriptions': descriptions_list,
        'locations': locations_list,
        'designation': designation_list,
        'variety': variety_list,
        'winery': winery_list,
        'redOrWhite': redOrWhite_list,
        'dryOrSweet': dryOrSweet_list,
        'bodied': bodied_list,
        'crispOrSmooth': crispOrSmooth_list,
        'tastesLike': tastesLike_list,
        'foodPairing': foodPairing_list   
    }
    return Labelslist

test = [
    "This red wine from pairs well with steak."
]
test_country = "Napa Valley"

def find_substring_positions(main_string, substring, label):
    start_positions = [start_pos for start_pos in range(len(main_string)) if main_string.startswith(substring, start_pos)]
    end_positions = [start_pos + len(substring) for start_pos in start_positions]
    return list(zip(start_positions, end_positions, label))

def createInputData():
    # Loop through the 'descriptions' list
    for description in labelsList['descriptions']:
        return description


def test():
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

labelsList = readCSV()

createInputData()