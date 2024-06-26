from transformers import AutoTokenizer
import argparse
from argparse import ArgumentParser
import json
from datasets import Dataset
import multiprocessing
import functools
import torch
import os
import pandas as pd

def check_args(config, dataset_path_through_command):
    """Check the configurations"""
    # REQUIRED configs
    if 'mode' not in config:
        raise Exception('Please provide the mode of the run. Choose between `train` & `eval`.')
    if 'model_id' not in config:
        raise Exception('Please provide the model_id provide in huggingface models')
    #if 'dataset_path' not in config:
    #    raise Exception('Please provide the dataset path that contains train.json & eval.json')
    if 'epochs' not in config:
        raise Exception('Please provide the epoch of the training data')
    
    # DEFAULT values for other configs
    if 'repository_id' not in config:
        config.repository_id = None # Hugging Face Repository id for uploading models
    if 'per_device_train_batch_size' not in config:
        config.per_device_train_batch_size = 4 # Batch size to use for training.
    if 'per_device_eval_batch_size' not in config:
        config.per_device_eval_batch_size = 1 # Batch size to use for testing.
    if 'max_input_length' not in config:
        config.max_input_length = 768 # Maximum length to use for generation
    if 'max_output_length' not in config:
        config.max_output_length = 256 # Maximum length to use for generation
    if 'generation_num_beams' not in config:
        config.generation_num_beams = 1 # Number of beams to use for generation
    if 'lr' not in config:
        config.lr = 1e-4 # Learning rate to use for training.
    if 'seed' not in config:
        config.seed = 42 # Random seed for all things random
    if 'deepspeed' not in config:
        config.deepspeed = "deepspeed_configs/z3_bf16.json" # Directory to the deepspeed configuration. Details in https://www.deepspeed.ai/tutorials/zero/
    if 'gradient_checkpointing' not in config:
        config.gradient_checkpointing = True # Whether to use gradient checkpointing. 
    if 'bf16' not in config:
        config.bf16 = True if torch.cuda.get_device_capability()[0] == 8 else False # Whether to use bf16.
    if 'num_workers' not in config:
        config.num_workers = multiprocessing.cpu_count()
        
    if dataset_path_through_command != None:
        config.dataset_path = dataset_path_through_command
    return config

def preprocess_function(examples, config, tokenizer, padding): 
    model_inputs = tokenizer(examples['source'], max_length=config.max_input_length, padding=padding, truncation=True)
    # Setup the tokenizer for targets
    #with tokenizer.as_target_tokenizer():
    labels = tokenizer(examples['target'], max_length=config.max_output_length, padding=padding, truncation=True)
    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def load_train_dataset(dataset, config, tokenizer, mode):
    tmp_dict = {"source":[],"target":[]}
    for idx, row in dataset.iterrows():
        tmp_dict['source'].append(str(row['input']))
        tmp_dict['target'].append(str(row['output']))
    dataset = Dataset.from_dict(tmp_dict)
    targetting_pder_device_batch_size = config.per_device_train_batch_size if mode == 'train' else config.per_device_eval_batch_size
    dataset = dataset.map(
        functools.partial(preprocess_function, config=config, tokenizer=tokenizer, padding='max_length'),
        batched=True,
        batch_size = targetting_pder_device_batch_size,
        num_proc=config.num_workers,
    )
    return dataset

def load_eval_dataset(dataset, config, tokenizer):
    tmp_dict = {"source":[],"target":[], "labels_list":[]}
    for idx, row in dataset.iterrows():
        tmp_dict['source'].append(str(row['input']))
        tmp_dict['target'].append(str(row['output']))
        tmp_dict['labels_list'].append(row['choices'].split('|||'))
    dataset = Dataset.from_dict(tmp_dict)
    dataset = dataset.map(
        functools.partial(preprocess_function, config=config, tokenizer=tokenizer, padding='max_length'),
        batched=True,
        batch_size=config.per_device_eval_batch_size,
        num_proc=config.num_workers
    )
    return dataset

parser = ArgumentParser()
parser.add_argument('--config', default=None, type=str)
parser.add_argument('--dataset_path', default=None, type=str)
arg_, _ = parser.parse_known_args()
if arg_.config is None:
    raise NameError("Include a config file in the argument please.")
config_path = arg_.config
with open(config_path) as config_file:
    config = json.load(config_file)
config = check_args(argparse.Namespace(**config), arg_.dataset_path)

tokenizer = AutoTokenizer.from_pretrained(config.model_id)

# Load train & eval datasets

if 'inference' in config.dataset_path:
    if 'rouge' in config_path:
        dev_file = f"{config.dataset_path}.csv"
        dev_csv_file = pd.read_csv(dev_file)
        #why load_train_dataset? load_eval_dataset?
        dev_dataset = load_train_dataset(dev_csv_file, config=config, tokenizer = tokenizer, mode='dev')
        #dev_dataset.save_to_disk(os.path.join(config.dataset_path, "dev"))
        dev_dataset.save_to_disk(os.path.join(config.dataset_path))
    else:
        test_file = f"{config.dataset_path}.csv"
        test_csv_file = pd.read_csv(test_file)
        test_dataset = load_eval_dataset(test_csv_file, config=config, tokenizer = tokenizer)
        test_dataset.save_to_disk(config.dataset_path)
else:
    train_file = f"{config.dataset_path}/train.csv"
    train_csv_file = pd.read_csv(train_file)
    train_dataset = load_train_dataset(train_csv_file, config=config, tokenizer = tokenizer, mode='train')
    train_dataset.save_to_disk(os.path.join(config.dataset_path,"train"))

print(config.dataset_path, 'done! :)')