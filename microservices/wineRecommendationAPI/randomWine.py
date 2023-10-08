import pandas as pd

def random():
    # Load your wine dataset
    wine_data = pd.read_csv("./data/winemag-data-130k-v2.csv")
    # Get a random row
    random_row = wine_data.sample(n=1)

    # Convert the random row to a JSON string
    random_row_json = random_row.to_json(orient='records')

    return random_row_json