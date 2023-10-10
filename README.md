# Sommelier
A wine recommendation model using natural language processing (NLP)

Inspired by the film "Sideways (2004)" I decided to develop a model designed to recommend a wine to a user based on descriptions akin to those provided by a sommelier following a tasting experience. The user can also specify any varieties, wineries, and region preferences that they would like their recommended wine to have as well as any tastes and food pairings. Due to this using Named Entity Recognition (NER), the user can ask questions that they normally would at a restuarant to discover their chosen wine.

### Questions you can ask:

Wine from a specific country
`Give me a wine from Italy`

A specific grape variety
`I'm looking for a Pinot Noir`

Wine from a country with a food pairing
`Give me a wine from Italy that pairs well with seafood`

A wine that has specific tastes
`Give me a rich and fruity red wine with notes of cherry and oak`

Can You Recommend a Wine for [Specific Dish or Cuisine]?
`Can you recommend a wine to pair with a salmon dish?`

Wine that has a region and food pairing
`Give me a wine from Italy that pairs with seafood`

Questions can also be asked regarding wines that are red or white, dry or sweet, light or full bodied, or crisp or smooth 

A custom Named Entity Recognition (NER) Model has been trained and customised to identify specific entities (in this case being wine regions, designations, varieties, wineries, general desriptions and food pairings). Whilst the default SpaCy model of en_core_web_lg is great, it struggles with defining multiple word locations such as Napa Valley, as well as complex wine varieties. Due to wine descriptions utilising a wide range of vocabulary, I decided it was important to refine the model further and so a custom model was created. The training takes place on 2000 entities which are individually labelled and takes around an hour to fully train. This is then tested on a dataset of around 100,000 entities.

## How to install:

### Pre-requisites

1. Go to the [Kaggle Wine Reviews Dataset](https://www.kaggle.com/datasets/zynicide/wine-reviews), make an account and download the dataset
2. Once you have downloaded the file titled 'winemag-data-130k-v2.csv', place it within the microservices/wineRecommendationAPI/data folder

#### Local Hosting

1. Clone the repo to a folder directory of your choosing
2. Place the file downloaded from the pre-requisite step into the microservices/wineRecommendationAPI/data directory if you haven't already
3. Open two terminals and cd into the repo within the folder directory
4. In terminal A, cd into the sommelier folder `cd sommelier`
5. In terminal B, cd into the microservices/wineRecommendationAPI folder `cd microservices/wineRecommendationAPI`
6. In both terminals run `pip install -r requirements.txt`
7. In terminal A, run `python manage.py collectstatic`, `python manage.py makemigrations`, `python manage.py migrate`, `python manage.py runserver`, and accept the 'yes' prompts
8. In terminal B, run `uvicorn wine_api:app --host 0.0.0.0 --port 8001`

#### Docker Container

1. Clone the repo to a folder directory of your choosing
2. Place the file downloaded from the pre-requisite step into the microservices/wineRecommendationAPI/data directory if you haven't already
3. Open a terminal and cd into the repo within the folder directory
4. Run `docker-compose build`
5. Run `docker-compose up`

## How to use

1. Open your browser to either your localhost(if you built locally) or your docker IP followed by the port 8000. E.g. `localhost:8000`
2. Type in a phrase to get a wine recommendation e.g. `A wine from Italy that pairs well with steak`
3. Your results will be visible upon clicking on 'recommend', alternatively you can click on the random wine button

## How to train

1. I have personally trained the model already on 2000 training entries, however if you wish to train on more entries then you can either run the python file 'trainCustomModel.py' locally or you can access 'http://localhost:8001/train' within the browser upin running the webserver. Don't forget to place the Kaggle dataset file within the `microservices/wineRecommendationAPI/data` first

## Methodology

To train a custom NER model using SpaCy, the following steps were taken:

a. Annotation of the wine dataset with entity labels; wine colour, dry, sweet, bodied, crisp, smooth, flavour and food pairing.

b. Training of the NER model on the annotated data to recognise these entities.

c. Fine-tuning the model to improve its accuracy.

Model Evaluation based on 70/30 Train/Test Split

| Metric    | Score |
|-----------|-------|
| Precision |99.56% |
| Recall    |99.56% |
| F1        |99.56% |

## Acknowledgements

A big thanks to Zack Thoutt for providing further inspiration and a test datatset.
