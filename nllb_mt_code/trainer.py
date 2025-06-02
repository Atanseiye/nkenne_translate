from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, Dataset, DatasetDict
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from transformers.utils import send_example_telemetry
from transformers import AutoTokenizer
from config import configuration
from evaluate import load
import pandas as pd
import numpy as np
import os

send_example_telemetry("translation_notebook", framework="pytorch")


# Define the folder path
folder_path = 'Nkenne/nllb'
lang_path = os.path.join(folder_path, f"{configuration['source_lang']}_to_{configuration['target_lang']}")
test_path = os.path.join(lang_path, 'test') 
testModel_path = os.path.join(lang_path, 'test/model') 
model_path = os.path.join(lang_path, 'model')
checkpoints = os.path.join(lang_path, 'model/checkpoints')
#preprocesses_path = os.path.join(folder_path, 'preprocesses')
data_path = os.path.join(lang_path, 'data')

#############################################
# Create the folder if it doesn't exist
#############################################
os.makedirs(folder_path, exist_ok=True)
os.makedirs(data_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)
os.makedirs(testModel_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)
#os.makedirs(preprocesses_path, exist_ok=True)
print(f"Folder created at: {folder_path}")

#############################################
# Connect to MongoDB and fetch data
#############################################

# MongoDB connection parameters
uri = configuration['mongo_uri']

def connect_to_mongo(uri, configuration):
    df = None
    # Connect to the MongoDB cluster
    client = MongoClient(uri, server_api=ServerApi('1'))
    try:
        # Send a ping to confirm a successful connection
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")

        # Access the database and collection
        database = client.get_database("nkenne-ai")

        # Query the collection
        data_collection = database[configuration['collection_name']]  # Use the collection name from the configuration

        query = configuration['query'] if 'query' in configuration else {}  # Use the query from the configuration
        dataset = list(data_collection.find(query))
        # Convert the dataset to a pandas DataFrame
        df = pd.DataFrame(dataset)[[configuration['target_lang'], configuration['source_lang']]]  # get all the data

        # Randomly divinde the dataset to train and test sets
        df_train_set = df.sample(frac=0.9, random_state=42) #radomly select 90% of the data, seed is set to 42
        remaining = df.drop(df_train_set.index) #drop the 80% selected from the data to get the remaining 20%
        # divide the testset into validation and test sets
        df_val_set = remaining.sample(frac=0.5, random_state=42)
        df_test_set = remaining.drop(df_val_set.index)
        # Reset the index of the dataframes
        df_train_set = df_train_set.reset_index(drop=True)
        df_test_set = df_test_set.reset_index(drop=True)
        df_val_set = df_val_set.reset_index(drop=True)
        print("Data loaded successfully")
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
    finally:
        # Close the connection
        client.close()
        print("MongoDB Connection closed")
    # TODO: modify return to return train and eval splits
    return df_train_set, df_test_set, df_val_set

# #####################################
# Connect to MongoDB and fetch data
# #####################################
df_train_set, df_test_set, df_val_set = connect_to_mongo(uri, configuration)
# Check if the data is loaded
if df_train_set is not None:
    print("Data loaded successfully")
print(f"Number of rows in the training set: {len(df_train_set)}")
print(f"Number of rows in the test set: {len(df_test_set)}")
print(f"Number of rows in the validation set: {len(df_val_set)}")


#############################################
# Drop nan values
#############################################
train_data = df_train_set.dropna().reset_index(drop=True)
test_data = df_test_set.dropna().reset_index(drop=True)
val_data = df_val_set.dropna().reset_index(drop=True)

#############################################
# Convert pandas DataFrame to Hugging Face Dataset
#############################################
train_data = Dataset.from_pandas(train_data)
test_data = Dataset.from_pandas(test_data)
val_data = Dataset.from_pandas(val_data)

#############################################
# Convert Dataset to pandas DataFrame
#############################################
train_data = train_data.to_pandas()
test_data = test_data.to_pandas()
val_data = val_data.to_pandas()

#############################################
# Convert to DatasetDict
#############################################
raw_datasets = DatasetDict({
    "train": Dataset.from_pandas(train_data),
    "test": Dataset.from_pandas(test_data),
    'validation': Dataset.from_pandas(val_data)
})

#############################################
# Convert DataFrame to the desired structure
#############################################
def convert_to_translation_dict(df):
    return {'translation': [{configuration['source_lang_']: row[configuration['source_lang']], configuration['target_lang_']: row[configuration['target_lang']]} for _, row in df.iterrows()]}

#############################################
# Convert each split to the desired structure
#############################################
train_data = convert_to_translation_dict(train_data)
val_data = convert_to_translation_dict(val_data)
test_data = convert_to_translation_dict(test_data)

#############################################
# Create DatasetDict with the new structure
#############################################
raw_datasets = DatasetDict({
    "train": Dataset.from_dict(train_data),
    "validation": Dataset.from_dict(val_data),
    "test": Dataset.from_dict(test_data)
})


#############################################
# Saving the data for future use
#############################################
# raw_datasets.save_to_disk(os.path.join(data_path))
# print(f"==> Data successfully saved to {os.path.join(data_path)}")

#############################################
# Load back the data
#############################################
# raw_datasets = DatasetDict.load_from_disk(os.path.join(data_path))
# print('==> Data successfully loaded back')

#############################################
# Load metrics
#############################################
metric = load("sacrebleu")

#############################################
# Load the base model
#############################################
# model_checkpoint = 'Helsinki-NLP/opus-mt-swc-en'
model_checkpoint ="facebook/nllb-200-distilled-600M"

#############################################
# Load the tokenizer
#############################################
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

#############################################
if "nllb" in model_checkpoint:
    # tokenizer.src_lang = "en-XX"
    # tokenizer.tgt_lang = "swc-SW"
    # Set source and target languages
    tokenizer.src_lang = "swa_Latn"  # Swahili (source language)
    tokenizer.tgt_lang = "en_Latn"  # English (target language)
#############################################

#############################################
# Add a prefix to help the model have more understanding of it's task
#############################################
if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = f"translate {configuration['source_lang']} to {configuration['target_lang']}: "
else:
    prefix = ""

#############################################
# Device Configuration
#############################################
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#############################################
# Preprocessing Function
#############################################
max_input_length = 128
max_target_length = 128
source_lang = configuration["source_lang_"]
target_lang = configuration["target_lang_"]

def preprocess_function(examples):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]

    # for input, target in zip(inputs, targets):
    #     print(f"Input: {input}")
    #     print(f"Target: {target}")
    #     print("===")
    #     break

    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    # with tokenizer.as_target_tokenizer():
    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


#############################################
# Function call to process the data
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
#############################################


#############################################
# Load the model
#############################################
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

#############################################
# Defining the training Arguments
#############################################
batch_size = 16
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_path}/", #nllb-finetuned-{source_lang}-to-{target_lang}",
    eval_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    run_name = f"latest_{configuration['lang']}_mod",
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
    report_to="wandb",  # Enables wandb logging
    logging_dir = os.path.join(
        model_path,  # Path to save the logs
        f"logs-{model_name}-{source_lang}-to-{target_lang}"),  # Unique name for the logs
    # Ensure the model saves to the correct directory
    save_strategy="epoch",  # Save the model at the end of each epoch
)


# args = Seq2SeqTrainingArguments(
#     f"{model_path}/",#nllb-finetuned-{source_lang}-to-{target_lang}",
#     eval_strategy = "steps",
#     learning_rate=wandb.config.learning_rate,
#     per_device_train_batch_size=wandb.config.batch_size,  # Use wandb config for batch size
#     per_device_eval_batch_size=wandb.config.batch_size,  # Use wandb config for eval batch size
#     weight_decay=wandb.config.weight_decay,  # Use wandb config for weight decay
#     predict_with_generate=True,  # To use generate to calculate generative metrics (ROUGE, BLEU)
#     save_total_limit=4,
#     num_train_epochs=1,
#     run_name = run_name,
#     fp16=True,
#     push_to_hub=True,
#     report_to="wandb",  # Enables wandb logging
#     logging_dir = os.path.join(
#         model_path,  # Path to save the logs
#         f"logs-{model_name}-{source_lang}-to-{target_lang}"),  # Unique name for the logs
#     # Ensure the model saves to the correct directory
#     save_strategy="steps",  # Save the model at the end of each epoch
#     save_steps=5000,  # Save every 500 steps
#     # eval_steps=50,  # Evaluate every 500 steps
#     # load_best_model_at_end=True,
#     # metric_for_best_model="eval_bleu",  # <-- Set your metric here
#     # greater_is_better=True,  
# )

#############################################
# Callingthe data collector
#############################################
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
model.to(device)

#############################################
# Post-processing and Metrics Computation
#############################################
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

#############################################
# The Training Argument
#############################################
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

#############################################
# Training the model
#############################################
trainer.train()

#############################################
# Saving the model
#############################################
trainer.save_model()



'''The End'''