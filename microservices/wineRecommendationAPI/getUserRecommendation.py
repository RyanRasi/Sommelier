import spacy
import pandas as pd
import re

def recommend(user_query):

    def extract_entities(text):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        # Extract nouns (potentially ingredients and fruits) and adjectives
        entities = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"] or (token.pos_ == "ADJ" in token.text.lower())]
        entities = " ".join(entities)
        return entities

    # Load the custom spaCy model
    nlp = spacy.load("./assets/wine_ner_model")

    # Load your wine dataset
    wine_data = pd.read_csv("./data/winemag-data-130k-v2.csv")
    wine_data['designation'] = wine_data['designation'].apply(lambda x: ' '.join([word for word in str(x).split() if len(word) > 1]))
    wine_data['variety'] = wine_data['variety'].apply(lambda x: ' '.join([word for word in str(x).split() if len(word) > 1]))
    wine_data['winery'] = wine_data['winery'].apply(lambda x: ' '.join([word for word in str(x).split() if len(word) > 1]))

    user_query = user_query.lower()

    user_query = user_query.replace('  ', ' ')
    print("User Query is: ", user_query)

    # Initialize variables to capture query details
    redOrWhite = None
    dryOrSweet = None
    bodied = None
    crispOrSmooth = None
    tastesLike = []
    foodPairing = []  

    # Filter wines based on user criteria
    filtered_wines = wine_data.copy()
    last_filtered_wines = filtered_wines

    # Check if region is mentioned
    set_region = False
    regions = set(sorted(set(wine_data[['country', 'province', 'region_1', 'region_2']].stack().dropna().str.lower()), key=len, reverse=True))
    # Specify columns to search
    for region in regions:
        if re.search(r'\b' + re.escape(region) + r'\b', user_query):
            set_region = region
            user_query = user_query.replace(region, '')
            #user_query = user_query.replace('  ', '')
    if set_region:
        columns_to_search = ["country", "region_1", "region_2", "province"]
        # Create a mask for each column
        masks = [wine_data[col].str.lower().eq(set_region) for col in columns_to_search]
        # Combine masks with logical OR to find rows where the entity is present in any of the specified columns
        combined_mask = pd.concat(masks, axis=1).any(axis=1)
        # Filter the DataFrame based on the combined mask
        filtered_wines = wine_data[combined_mask]
    if not filtered_wines.empty:
        last_filtered_wines = filtered_wines
    else:
        filtered_wines = last_filtered_wines
    print(set_region)

    # Check if designation is mentioned
    set_designation = False
    designations = set(sorted(set(wine_data[['designation']].stack().dropna().str.lower()), key=len, reverse=True))
    for designation in designations:
        if re.search(r'\b' + re.escape(designation) + r'\b', user_query):
            set_designation = designation
            user_query = user_query.replace(designation, '')
            ##      
    if set_designation:
        filtered_wines = filtered_wines[filtered_wines["designation"].str.lower() == set_designation.lower()]
    if not filtered_wines.empty:
        last_filtered_wines = filtered_wines
    else:
        filtered_wines = last_filtered_wines
    print(set_designation)

    # Check if variety is set
    set_variety = False
    varieties = set(sorted(set(wine_data[['variety']].stack().dropna().str.lower()), key=len, reverse=True))
    for variety in varieties:
        if re.search(r'\b' + re.escape(variety) + r'\b', user_query):
            set_variety = variety
            user_query = user_query.replace(variety, '')
            #user_query = user_query.replace('  ', '')      
    if set_variety:
        filtered_wines = filtered_wines[filtered_wines["variety"].str.lower() == set_variety.lower()]
    if not filtered_wines.empty:
        last_filtered_wines = filtered_wines
    else:
        filtered_wines = last_filtered_wines
    print(set_variety)

    # Check if winery is set
    set_winery = False
    wineries = set(sorted(set(wine_data[['winery']].stack().dropna().str.lower()), key=len, reverse=True))
    for winery in wineries:
        if re.search(r'\b' + re.escape(winery) + r'\b', user_query):
            set_winery = winery
            user_query = user_query.replace(winery, '')
            #user_query = user_query.replace('  ', '')      
    if set_winery:
        filtered_wines = filtered_wines[filtered_wines["winery"].str.lower() == set_winery.lower()]
    if not filtered_wines.empty:
        last_filtered_wines = filtered_wines
    else:
        filtered_wines = last_filtered_wines
    print(set_winery)

    print(user_query)
    #user_query = user_query.replace(" ", "")

    user_query = extract_entities(user_query)
    print(user_query)
    # Process the user query
    doc = nlp(user_query)
    # Extract information from the query
    for ent in doc.ents:
        print(ent.label_, " - ", ent.text)
        if ent.label_ == 'COLOUR':
            redOrWhite = ent.text
        elif ent.label_ == 'DRYORSWEET':
            dryOrSweet = ent.text
        elif ent.label_ == 'BODIED':
            bodied = ent.text
        elif ent.label_ == 'CRISPORSMOOTH':
            crispOrSmooth = ent.text
        elif ent.label_ == 'FLAVOUR':
            tastesLike.append(ent.text)
        elif ent.label_ == 'FOOD':
            foodPairing.append(ent.text)

    # Others...
    if redOrWhite:
        filtered_wines = filtered_wines[filtered_wines["description"].str.lower().str.contains(redOrWhite.lower())]

    if not filtered_wines.empty:
        last_filtered_wines = filtered_wines
    else:
        filtered_wines = last_filtered_wines

    if dryOrSweet:
        filtered_wines = filtered_wines[filtered_wines["description"].str.lower().str.contains(dryOrSweet.lower())]

    if not filtered_wines.empty:
        last_filtered_wines = filtered_wines
    else:
        filtered_wines = last_filtered_wines

    if bodied:
        filtered_wines = filtered_wines[filtered_wines["description"].str.lower().str.contains(bodied.lower())]

    if not filtered_wines.empty:
        last_filtered_wines = filtered_wines
    else:
        filtered_wines = last_filtered_wines

    if crispOrSmooth:
        filtered_wines = filtered_wines[filtered_wines["description"].str.lower().str.contains(crispOrSmooth.lower())]

    if not filtered_wines.empty:
        last_filtered_wines = filtered_wines
    else:
        filtered_wines = last_filtered_wines

    if tastesLike:
        temp_filtered_wines = filtered_wines
        for eachtastesLike in tastesLike:
            filtered_wines = filtered_wines[filtered_wines["description"].str.lower().str.contains(eachtastesLike.lower())]
            if filtered_wines.empty:
                filtered_wines = temp_filtered_wines
            else:
                temp_filtered_wines = filtered_wines

    if not filtered_wines.empty:
        last_filtered_wines = filtered_wines
    else:
        filtered_wines = last_filtered_wines

    if foodPairing:
        temp_filtered_wines = filtered_wines
        for eachFoodPairing in foodPairing:
            filtered_wines = filtered_wines[filtered_wines["description"].str.lower().str.contains(eachFoodPairing.lower())]
            if filtered_wines.empty:
                filtered_wines = temp_filtered_wines
            else:
                temp_filtered_wines = filtered_wines

    if not filtered_wines.empty:
        last_filtered_wines = filtered_wines
    else:
        filtered_wines = last_filtered_wines

    # Provide wine recommendations based on the filtered wines
    if not filtered_wines.empty:
        recommendations = filtered_wines.head(1).to_json(orient='records')
        print("Wine Recommendations:")
        return recommendations
    else:
        print("No wines found matching the criteria.")
        return "No wines found matching the criteria."