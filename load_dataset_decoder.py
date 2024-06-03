from typing import Dict, Optional, Sequence, List
from torch.utils.data import Dataset
import transformers
import torch
from datasets import load_dataset, load_from_disk
from tqdm import tqdm 
import random
import time
import pandas as pd
import os

def token_function(datum, tokenizer, dataset_args, padding = 'max_length'):
    input_text = datum['source']
    output_text = datum['target']
    if output_text is None:
        output_text = 'do not have answer'
    elif type(output_text) == int or type(output_text) == float:
        output_text = str(output_text)   
    elif output_text.strip() == "":
        output_text = 'do not have answer'

    if input_text is None or input_text.strip() == "":
        print(datum)
        input_text = " "

    input_length, output_length = dataset_args['input_length'], dataset_args['output_length']
    max_length = dataset_args['max_length']
    
    context_ids = tokenizer(input_text).input_ids
    label_ids = tokenizer(output_text).input_ids

    if len(context_ids) + len(label_ids) > max_length-1:
        if len(label_ids) < output_length:
            context_ids = context_ids[-(max_length-1-len(label_ids)):]
        else:
            context_ids = context_ids[-(input_length-1):]
            label_ids = label_ids[:output_length]

    input_ids = context_ids + label_ids + [tokenizer.eos_token_id]
    label_ids = [-100]*len(context_ids) + label_ids + [tokenizer.eos_token_id]
    
    return {
        'input_ids': torch.LongTensor(input_ids),
        'labels': torch.LongTensor(label_ids)
    }


class SupervisedDataset(Dataset):
    def __init__(self, dataset_path: Sequence[str], args, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        if args.mode == 'train':
            self.dataset = load_from_disk(os.path.join(args.dataset_lists, "train"))
        else:
            self.dataset = load_from_disk(dataset_path)
        self.tokenizer = tokenizer
        self.dataset_args = {
            "input_length" : 768,
            "output_length" : 512,
            'max_length': 1280,
        }


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        datum = self.dataset[i]
        datum = token_function(datum, self.tokenizer, self.dataset_args)

        return datum


class DECODERDATASET(Dataset):
    def __init__(self, dataset_path, tokenizer, input_length, output_length):
        self.tokenizer = tokenizer

        self.dataset_name = dataset_path


        self.dataset = load_dataset('csv', data_files=dataset_path)['train']

        if self.dataset[0]['choices'] == None:
            self.generation = 1
        else:
            self.generation = 0
        
        # print(f'Length of dataset retrieving is.. {len(self.dataset)}')
        self.input_length = input_length
        self.output_length = output_length
        self.max_sequence_length = self.input_length + self.output_length

        self.bos_token_id = self.tokenizer.bos_token_id
        # self.bos_token_id = 3
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

    def __len__(self):
        return len(self.dataset)

    def pad_sequence_max(self, queries, padding_value):
        padded_tokens = [query_i + [padding_value] * (self.max_sequence_length - len(query_i)) for query_i in queries]
        
        return torch.LongTensor(padded_tokens).long()

    def pad_sequence_left(self, queries, padding_value):
        queries_len = [len(s) for s in queries]
        max_query_len = max(queries_len)
        padded_tokens = [[padding_value] * (max_query_len - len(query_i)) + query_i for query_i in queries]
        
        return torch.LongTensor(padded_tokens).long()

    def convert_to_feature_tokenizer(self, input_, target_, options):
        context_ids = self.tokenizer(input_).input_ids
        if len(context_ids) > self.input_length-1:
            context_ids = context_ids[-(self.input_length):]# + [self.bos_token_id]
        else:
            context_ids = context_ids# + [self.bos_token_id]

        input_ids=context_ids
        return input_ids

    def convert_to_features(self, example_batch, index):
        # prompt evaluation
        option_label = -1

        input_ = example_batch['input']
        target_ = example_batch['output']

        try:
            options = example_batch['choices'].split('|||')
            options = [op.strip() for op in options]
        except:
            options = None
        try:
            option_label = options.index(str(target_).strip())
        except:
            option_label = None
            
        input_ids = self.convert_to_feature_tokenizer(input_, target_, options)
        return input_ids, target_, options, option_label


    def __getitem__(self, index):
        indexed_data = self.dataset[index]    

        input_ids, target, options, option_label = self.convert_to_features(indexed_data, index)

        option_list = options


        return {"input_ids": input_ids, "target": target, "option_list": option_list, "option_label": option_label}

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        option_list = [ins["option_list"] for ins in batch]
        option_label = [ins["option_label"] for ins in batch]

        target = [ins["target"] for ins in batch]

        attention_mask = input_ids != self.pad_token_id

        return  {"input_ids": input_ids, "attention_mask":attention_mask, "target": target, "option_list": option_list, "option_label": option_label}
