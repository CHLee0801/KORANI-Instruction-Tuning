import json
from datasets import load_dataset
from scipy import stats
from transformers import AutoModelForCausalLM, set_seed, GenerationConfig, AutoTokenizer
from accelerate import Accelerator
import argparse
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import os
from load_dataset_decoder import DECODERDATASET
from third_party.trainers import ids_to_clean_text, _rougel_score, metric_rouge_english, metric_rouge_korean
import csv 
accelerator = Accelerator()

def prepare_option(batch,index,tokenizer):
    option= batch["option_list"][0][index]

    option_ids= tokenizer.encode(option)
    option_len=len(option_ids)
    pad_token_id = tokenizer.pad_token_id
    # option_ids=option_ids+[self.eos_token_id]
    input_ids=batch['input_ids'][:]
    label_ids=[]
    for i in range(len(input_ids)):
        input_ids[i]=torch.tensor(input_ids[i]+option_ids)
        label_ids.append(torch.tensor([-100]*(len(input_ids[i]) - len(option_ids)) + option_ids))
    
    input_ids = pad_sequence(input_ids, True, padding_value=pad_token_id).long()
    label_ids = pad_sequence(label_ids, True, padding_value=-100).long()
    
    # input_ids = torch.tensor(input_ids)
    input_mask = input_ids != pad_token_id
    label_mask = label_ids != -100

    return input_ids,label_ids,input_mask,label_mask,option_ids
    
def decoder_evaluate_prob(model, tokenizer, dataset_file, args):
    model.to('cuda')
    model.eval()    

    tokenizer.padding_side = 'left'
    
    accelerator.wait_for_everyone()
     
    dataloader = torch.utils.data.DataLoader(dataset_file, batch_size = args.per_device_eval_batch_size, collate_fn=dataset_file.collate_fn)
    
    dataloader = accelerator.prepare(dataloader)
    entries = []
    indx = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            prob_list = []
            for index in range(len(batch["option_list"][0])):
                
                input_ids,label_ids,input_mask,label_mask,option_ids=prepare_option(batch,index,tokenizer)

                outputs = model(
                    input_ids=input_ids.cuda(),
                    attention_mask=input_mask.cuda(),
                    labels=label_ids.cuda()
                    )
                logits=outputs.logits
                logits=logits[..., :-1, :].contiguous()
                label_mask = label_mask[..., 1:].contiguous()
                
                if logits.dtype!=torch.float32:
                    logits=logits.type(torch.float32)
                logits=logits[label_mask,:].reshape(logits.shape[0],-1,logits.shape[2])
                logits =torch.log_softmax(logits, dim=-1).cpu()
                seq_token_log_prob=torch.zeros(logits.shape[:-1])
                #print(seq_token_log_prob.shape, logits.shape, lm_labels.shape)
                for i in range(seq_token_log_prob.shape[0]):
                    for j in range(seq_token_log_prob.shape[1]):
                        seq_token_log_prob[i][j] = logits[i][j][option_ids[j]]
                seq_log_prob = seq_token_log_prob.sum(dim=-1)
                prob_list.append(seq_log_prob)

            input_list = tokenizer.batch_decode(batch['input_ids'])
            concat = torch.stack(prob_list)
            predictions = concat.argmax(dim=0)
            dec = [batch["option_list"][elem_num][i.item()] for elem_num, i in enumerate(predictions)]
            output_label = batch["option_label"]
            entries.append({'pred':predictions,'output_label':output_label, 'input_list':input_list})

    accelerator.wait_for_everyone()
    # print(len(entries))
    entries_gathered = accelerator.gather_for_metrics(entries)
    # print(len(entries_gathered))
    if accelerator.is_main_process:
        
        log_path = 'out/results.csv'
        f = open(log_path, 'a', encoding='utf-8', newline='')
        wr = csv.writer(f)
            
        acc, total_cnt = 0,0
        input_dict = {}
        preds, refs, input_list = [], [], []

        for output_result in entries_gathered:
            preds = preds + list(output_result['pred'])
            refs = refs + output_result['output_label']
            input_list = input_list + output_result['input_list']
        input_dict = {}
        for i in range(len(input_list)):
            if input_list[i] not in input_dict:
                input_dict[input_list[i]]= 1
            else:
                continue
            if preds[i] == refs[i]:
                acc += 1
            total_cnt += 1
        print('acc:', round(acc/total_cnt*100,4))
        if args.checkpoint_path == '':
            wr.writerow([args.model_id, args.current_dataset, round(acc/total_cnt*100,4)])
        else:
            wr.writerow([args.checkpoint_path, args.current_dataset, round(acc/total_cnt*100,4)])
        f.close()  

    # exit()    
    accelerator.wait_for_everyone()

def decoder_evaluate_generate(model, tokenizer, dataset_path, args):
    model.to('cuda')
    model.eval()    

    tokenizer.padding_side = 'left'
    
    accelerator.wait_for_everyone()

    #Loading the configs to use for generation
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        #eos_token_id=100_000,
        eos_token_id=tokenizer.eos_token_id,
        #temperature=args.decoding_temperature,
        top_k=0.0,
        top_p=1.0,
        #top_p=0.9,
        no_repeat_ngram_size=3,
        do_sample=True,
        num_return_sequences=1,
        max_new_tokens=256,
        min_new_tokens=2
    )           
    
    dataset_file = load_dataset('csv', data_files=dataset_path)['train']
    dataset_file = dataset_file.remove_columns("choices")
    dataloader = torch.utils.data.DataLoader(dataset_file, batch_size = args.per_device_eval_batch_size)
    
    dataloader = accelerator.prepare(dataloader)
    entries = []
    indx = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            input_entries = []
            for j in range(len(batch['input'])):
                input_entries.append(batch['input'][j])
            input_ids = tokenizer(input_entries, max_length=args.max_output_length, padding=True, truncation=True, return_tensors="pt").input_ids
            input_ids = input_ids.to('cuda')
            outputs = model.generate(
                input_ids = input_ids,
                generation_config = generation_config,
                max_new_tokens = args.max_output_length,
                return_dict_in_generate = True
            )
            outputs_text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            
            for j in range(len(input_entries)):
                # print(j)
                input_text = input_entries[j]
                output_text = str(outputs_text[j][len(input_text):])
                golden = str(batch['output'][j])
                
                # print(f'INPUT: {input_text}')
                # print(f'OUTPUT: ', output_text)

                # print(f'GOLDEN: ', golden)
                # print("==========")
                
                entry = {
                    "id": indx,
                    "input": input_text,
                    "output": output_text,
                    "golden": golden
                }
                entries.append(entry)
                indx+=1

    accelerator.wait_for_everyone()
    # print(len(entries))
    entries_gathered = accelerator.gather_for_metrics(entries)
    # print(len(entries_gathered))
    if accelerator.is_main_process:
        log_path = 'out/results.csv'
        f = open(log_path, 'a', encoding='utf-8', newline='')
        wr = csv.writer(f)
        
        em_cnt, total_cnt = 0,0
        input_dict = {}
        preds, refs = [], []
        for output_result in entries_gathered:
            if output_result['input'] not in input_dict:
                input_dict[output_result['input']] = 1
            else:
                continue
            # print(output_result)
            preds.append(output_result["output"])
            refs.append(output_result["golden"])
            total_cnt += 1
        
        result = metric_rouge_korean(preds, refs)    
        print('rouge:', result['rouge'])
        if args.checkpoint_path == '':
            wr.writerow([args.model_id, args.current_dataset, round(result['rouge']*100,3)])
        else:
            wr.writerow([args.checkpoint_path, args.current_dataset, round(result['rouge']*100,3)])
        f.close()  

    # exit()    
    accelerator.wait_for_everyone()
