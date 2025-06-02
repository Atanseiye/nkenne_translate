# nkenne_translate

## Installation

1. Clone the repository and install dependencies:

```bash
git clone https://github.com/Atanseiye/nkenne_translate.git
```

2. Navigate to the script directory
```bash
cd nllb_mt_code
```

3. Install the required dependecies
```bash
pip install -r requirements.txt  # or use npm install if it's JS
```

4. Open `config.py` and modify the settings accordingly
- To train swahili-to-english model.
```bash
name = "swahili"
```

- To train english-to-swahili model
```bash
name = "swahili"
'source_lang': f'english_{name}', #-----for MongoDB collection - collection list
'collection_name': 'english_text', #-----for loading data
```

- Update the database uri
```bash
'mongo_uri': '', #-----MongoDB URI
```

Change all parameter as necessary

## Start the training process
To initialise the training process
```bash
python run trainer.py
```
