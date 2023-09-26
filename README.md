# Sommelier
A wine recommendation model using natural language processing (NLP)

Inspired by the film "Sideways (2004)" I decided to create a model identifies the variety, winery, and the location of a wine based on a description that a sommelier could provide after tasting it but also provides wine recommendations and pairings based on food dishes.

A custom Named Entity Recognition (NER) Model is trained and customised to identify specific entities (in this case being wine varieties, regions, and food pairings). 

To train a custom NER model using spaCy, the following steps were taken:

a. Annotation of the wine dataset with entity labels (e.g., mark wine varieties, regions, and food pairings).

b. Training of the NER model on the annotated data to recognise these entities.

c. Fine-tuning the model to improve its accuracy.

Acknowledgements
A big thanks to Zack Thoutt for providing further inspiration and a test datatset.