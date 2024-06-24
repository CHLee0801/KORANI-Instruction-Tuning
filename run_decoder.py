import os
import argparse
from argparse import ArgumentParser
import json
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    set_seed,
)
from tqdm import tqdm
import csv
from datasets import Dataset, load_from_disk, load_metric
import torch
import evaluate
import nltk
import numpy as np
import multiprocessing
import wandb
from load_dataset_decoder import SupervisedDataset, DECODERDATASET
from huggingface_hub import HfFolder
from transformers import TrainingArguments
from third_party.trainers import PostProcessor
from setproctitle import *
from transformers.trainer import Trainer
from third_party.trainers import ids_to_clean_text, _rougel_score, metric_rouge_english, metric_rouge_korean
from evaluate_decoder import decoder_evaluate_prob, decoder_evaluate_generate

setproctitle('MT Experiment')

local_rank = int(os.getenv('LOCAL_RANK', '0'))
os.environ["WANDB_DISABLED"] = "true"

def check_args(config, checkpoint_path_through_command, dataset_path_through_command, target_cluster, target_dataset):
    """Check the configurations"""
    # REQUIRED configs
    if 'mode' not in config:
        raise Exception('Please provide the mode of the run. Choose between `train` & `eval`.')
    if 'model_id' not in config:
        raise Exception('Please provide the model_id provide in huggingface models')
    #if 'dataset_path' not in config and dataset_path_through_command == None:
    #    raise Exception('Please provide the dataset path that contains train.json & eval.json')
    if 'epochs' not in config:
        raise Exception('Please provide the epoch of the training data')
    if 'output_dir' not in config:
        raise Exception('Please provide the output directory to save the log files & model checkpoint')
    
    # DEFAULT values for other configs
    if 'per_device_train_batch_size' not in config:
        config.per_device_train_batch_size = 1 # Batch size to use for training.
    if 'per_device_eval_batch_size' not in config:
        config.per_device_eval_batch_size = 1 # Batch size to use for testing.
    if 'max_input_length' not in config:
        config.max_input_length = 768 # Maximum length to use for generation
    if 'max_output_length' not in config:
        config.max_output_length = 256 # Maximum length to use for generation
    if 'generation_num_beams' not in config:
        config.generation_num_beams = 1 # Number of beams to use for generation
    if 'lr' not in config:
        config.lr = 1e-5 # Learning rate to use for training.
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
    if 'gradient_accumulation_steps' not in config:
        config.gradient_accumulation_steps = 1
    if 'dataset_length' not in config:
        config.dataset_length = 100000
    if 'n_gpu' not in config:
        config.n_gpu = 16
    if 'checkpoint_path' not in config:
        config.checkpoint_path = ""
    
    if checkpoint_path_through_command != None:
        config.checkpoint_path = checkpoint_path_through_command
    if dataset_path_through_command != None:
        config.dataset_path = dataset_path_through_command
    if target_cluster != None:
        config.target_cluster = target_cluster
    else:
        config.target_cluster = None
    if target_dataset != None:
        config.target_dataset = target_dataset
    else:
        config.target_dataset = None
    # etc.
    if 'repository_id' not in config:
        config.repository_id = None # Hugging Face Repository id for uploading models
    if 'hf_token' not in config:
        config.hf_token = HfFolder.get_token() # Token to use for uploading models to Hugging Face Hub.
    if 'wandb' not in config:
        config.wandb =  False
    if 'wandb_entity' not in config:
        config.wandb_entity = 'changholee' # Default wandb entity to log experiments to. Change with your wandb entity
    if 'wandb_project' not in config:
        config.wandb_project = 'mt5_flm' # Change depending on your project name
    if 'wandb_run_name' not in config and config.wandb == True:
        if config.checkpoint_path != "":
            config.wandb_run_name = config.checkpoint_path + '-' + config.dataset_path # Provide name to the run
        else:
            config.wandb_run_name = config.dataset_path

    return config

nltk.download("punkt", quiet=True)

# Metric
#metric = evaluate.load("rouge")
metric = evaluate.load("accuracy")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def training_run(args):
    # Set Random Seed :)
    set_seed(args.seed)

    eval_path_lists = []
    
    if not os.path.exists('KORANI'):
        print("If you want to evaluate on KORANI, please download the dataset!")
    else:
        korani_cluster_list = os.listdir('KORANI')
        korani_dataset_dict = {}
        for korani_cluster in korani_cluster_list:
            dataset_list = os.listdir(f'KORANI/{korani_cluster}')
            for dataset in dataset_list:
                korani_dataset_dict[dataset] = korani_cluster
        if args.target_cluster == None and args.target_dataset == None:
            for dataset in korani_dataset_dict:
                test_list = os.listdir(f'KORANI/{korani_dataset_dict[dataset]}/{dataset}')
                eval_path_lists += [f'KORANI/{korani_dataset_dict[dataset]}/{dataset}/{test_tmp}' for test_tmp in test_list if 'train' not in test_tmp]
        if args.target_cluster != None: 
            if ',' in args.target_cluster:
                cluster_lists = args.target_cluster.split(',')
            else:
                cluster_lists = [args.target_cluster]
            for cluster in cluster_lists:
                if cluster.strip() not in korani_cluster_list:
                    raise Exception('Invalid cluster type!')
                dataset_lists = os.listdir(f'KORANI/{cluster}')
                for dataset in dataset_lists:
                    test_list = os.listdir(f'KORANI/{cluster}/{dataset}')
                    eval_path_lists += [f'KORANI/{cluster}/{dataset}/{test_tmp}' for test_tmp in test_list if 'train' not in test_tmp]
        if args.target_dataset != None:
            if ',' in args.target_dataset:
                dataset_lists = args.target_dataset.split(',')
            else:
                dataset_lists = [args.target_dataset]
            for dataset in dataset_lists:
                if dataset.strip() not in korani_dataset_dict:
                    raise Exception('Invalid dataset type!')
                test_list = os.listdir(f'KORANI/{korani_dataset_dict[dataset]}/{dataset}')
                test_tmp_list = [f'KORANI/{korani_dataset_dict[dataset]}/{dataset}/{test_tmp}' for test_tmp in test_list if 'train' not in test_tmp]
                for test_tmp in test_tmp_list:
                    if test_tmp in eval_path_lists:
                        continue
                    else:
                        eval_path_lists.append(test_tmp)
    
    eval_path_lists.sort()
    # Load model & tokenizer from huggingface
    if args.checkpoint_path != "":
        if local_rank == 0:
            print(args.checkpoint_path)
        model = AutoModelForCausalLM.from_pretrained(
            #args.model_id,
            args.checkpoint_path,
            use_cache=False if args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16
        )    
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            use_cache=False if args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16
        )   

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    train_dataset, eval_datasets = [],[]

    # Load train & eval datasets
    if args.mode == 'train':
        train_dataset = SupervisedDataset(args.dataset_lists, args, tokenizer)
        # train_dataset = load_from_disk(os.path.join(args.dataset_lists, "train"))
    else:
        for path in eval_path_lists:
            dataset_file = DECODERDATASET(dataset_path=path, tokenizer=tokenizer, input_length=768, output_length=256)
            generation_or_not = dataset_file.generation
            eval_datasets.append([dataset_file, generation_or_not])
        
    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    def get_accuracy(preds, labels):
        total_cnt = 0
        correct = 0
        for i in range(len(preds)):
            total_cnt+=1
            if preds[i] == labels[i]:
                correct+=1
        return {'accuracy': correct / total_cnt}

    # Define compute metrics function
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        post_processor = PostProcessor(tokenizer, ignore_pad_token_for_loss=True)
        decoded_preds, decoded_labels = post_processor.process(preds, labels)

        if args.current_generation_or_not==1:
            print("rouge!")
            result = metric_rouge_korean(decoded_preds, decoded_labels)    
        else:
            args.metric = get_accuracy(preds=decoded_preds, labels=decoded_labels)

        result = {k: round(v * 100, 4) for k, v in result.items()}
        if local_rank == 0:
            print(result)
            log_path = 'out/results.csv'
            f = open(log_path, 'a', encoding='utf-8', newline='')
            wr = csv.writer(f)
            if args.current_generation_or_not == 1:
                if args.checkpoint_path == '':
                    wr.writerow([args.model_id, args.current_dataset, result['rouge']])
                else:
                    wr.writerow([args.checkpoint_path, args.current_dataset, result['rouge']])
                f.close()  
            else:
                if args.checkpoint_path == '':
                    wr.writerow([args.model_id, args.current_dataset, result['accuracy']])
                else:
                    wr.writerow([args.checkpoint_path, args.current_dataset, result['accuracy']])
                f.close()  
        return result

    # Define training args
    # output_dir = args.repository_id if args.repository_id else args.model_id.split("/")[-1]

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        # predict_with_generate=True,
        # generation_max_length=args.max_output_length,
        # generation_num_beams=args.generation_num_beams,
        fp16=False,  # T5 overflows with fp16
        bf16=args.bf16,  # Use BF16 if available
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        deepspeed=args.deepspeed,
        #sharded_ddp=True,
        gradient_checkpointing=args.gradient_checkpointing,
        # logging & evaluation strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=1000,
        evaluation_strategy="no",
        # save_steps=ckpt_saving_steps,
        save_strategy="epoch",
        save_total_limit=3,
        # load_best_model_at_end=True,
        # push to hub parameters
        # report_to="wandb",
        push_to_hub=True if args.repository_id else False,
        hub_strategy="every_save",
        hub_model_id=args.repository_id if args.repository_id else None,
        hub_token=args.hf_token,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # adafactor=True
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    if args.mode=='train':
        print('Starting Training!')
        trainer.train()

        # Save our tokenizer and create model card
        tokenizer.save_pretrained(output_dir)
        trainer.create_model_card()

        # Push the results to the hub
        if args.repository_id:
            trainer.push_to_hub()
    elif args.mode=='eval':
        if local_rank == 0:
            print('Starting Evaluation!')
        
        for idx, eval_dataset in enumerate(eval_datasets):
            args.current_dataset = eval_path_lists[idx]
            if local_rank == 0:
                print(args.current_dataset)
            if eval_dataset[1] == 0:
                decoder_evaluate_prob(model, tokenizer, eval_dataset[0], args)
            else:
                decoder_evaluate_generate(model, tokenizer, args.current_dataset, args)
    else:
        raise Exception('Currently only supporting train & eval.')

def main():
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    parser.add_argument('--checkpoint_path', default=None, type=str)
    parser.add_argument('--dataset_path', default=None, type=str)
    parser.add_argument('--target_cluster', default=None, type=str)
    parser.add_argument('--target_dataset', default=None, type=str)
    arg_, _ = parser.parse_known_args()

    if arg_.config is None:
        raise NameError("Include a config file in the argument please.")
    config_path = arg_.config
    with open(config_path) as config_file:
        config = json.load(config_file)
    config = check_args(argparse.Namespace(**config), arg_.checkpoint_path, arg_.dataset_path, arg_.target_cluster, arg_.target_dataset)

    if config.wandb:
        wandb.init(entity=config.wandb_entity, project=config.wandb_project, name=config.wandb_run_name)
    training_run(config)

if __name__ == "__main__":
    main()