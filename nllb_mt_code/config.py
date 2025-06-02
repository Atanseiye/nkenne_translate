# This file contains the configuration for the application.
# It includes the paths to the models, data, and other resources.
# It also includes the MongoDB URI for storing and retrieving data.
# The configuration is used by the application to load the models and data.
# The configuration is stored in a dictionary and can be accessed by the application.

name = 'yoruba' #-----Change this to the name of the source language

configuration = {
    'lang': name, #-----Change this to the name of the source language
    'collection_name': f'english_{name}', #-----for MongoDB collection - collection list
    'source_lang': 'english_text', #-----for loading data
    'target_lang': f'{name}_text', #-----for loading data
    'query': {'file_url':"https://opus.nlpl.eu/results/en&sw/corpus-result-table"}, #-----for MongoDB query to fetch the data
    'source_lang_': 'swc', #-----as metadata
    'target_lang_': 'en', #-----as metadata
    'mongo_uri': 'mongodb+srv://kolade:bM0MkpVxYu9qMhAc@nkenne-cluster.zbsi9.mongodb.net/', #-----MongoDB URI
}