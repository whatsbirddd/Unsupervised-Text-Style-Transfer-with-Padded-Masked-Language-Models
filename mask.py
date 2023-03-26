from glob import glob
import os
import json
import numpy as np
import pandas as pd
import argparse
import yaml
import torch
import random
from time import time
from typing import Optional
import copy

from tqdm.auto import tqdm
from tqdm import notebook

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import transformers
from transformers import BertConfig, BertTokenizer, BertForMaskedLM

def get_text(file_path):
    source_file = json.load(open(file_path, "r"))
    title = source_file['sourceDataInfo']['newsTitle']
    content = source_file['sourceDataInfo']['newsContent']
    text = title + '[SEP]' + content
    return text

def get_logit(file_path, cfg, model, tokenizer, span_size, SOURCE = None):
    text = get_text(file_path)
    input = tokenizer(text, 
                      max_length = cfg["MODEL"]["max_seq_length"], 
                      padding = "max_length", 
                      truncation = True, 
                      return_tensors="pt"
                      )
    if SOURCE :
        token_type_ids = torch.tensor([0] * cfg["MODEL"]["max_seq_length"], dtype = torch.long) #source
    else:
        token_type_ids = torch.tensor([1] * cfg["MODEL"]["max_seq_length"], dtype = torch.long) #target
    input["token_type_ids"] = token_type_ids
    with torch.no_grad():
        output = model(**input).logits
    indices = input.input_ids.unsqueeze(axis=-1) #(1, 512, 1)
    logit_of_input_ids = torch.gather(output, 2, indices).squeeze() #(1, 512, 1) : torch.gather 좋네

    ## input에서 [sep]의 index찾기
    source_sep_id = (input.input_ids[0] == tokenizer.sep_token_id).nonzero().squeeze()[0] 
    ## [sep]나오기 전까지 span길이만큼의 logit 합으로 구성된 matrix구하기
    n_gram_logits = torch.tensor([sum(logit_of_input_ids[i : i+span_size]) for i in range(0, source_sep_id - span_size + 1)])
    return n_gram_logits

def run(file_path, cfg, model, tokenizer, span_size=4):
    s_n_gram_logits = get_logit(file_path, cfg, model, tokenizer, span_size=span_size, SOURCE = True)
    t_n_gram_logits = get_logit(file_path, cfg, model, tokenizer, span_size=span_size, SOURCE = False)
    
    # span의 logit 차이가 큰 index부분 찾기 -> MASK할 부분
    diff = s_n_gram_logits-t_n_gram_logits
    mask_idx = diff.argmax() #source index로 사용하면 됨.

    text = get_text(file_path) 
    label = tokenizer(text, 
                      max_length = cfg["MODEL"]["max_seq_length"], 
                      padding = "max_length", 
                      truncation = True, 
                      return_tensors = "pt"
                      )
    masked_input = copy.deepcopy(label)
    masked_input['input_ids'][0, mask_idx : mask_idx+span_size] = tokenizer.mask_token_id
    masked_input['token_type_ids'] = torch.tensor([1] * cfg["MODEL"]["max_seq_length"], dtype = torch.long) #target
    
    return masked_input, label

def save(input_list, label_list, savedir): 
    input_dict = {}
    for i, input in enumerate(tqdm(input_list)):
        if len(input_dict) == 0:
            for k in input.keys():
                input_dict[k] = []
        
        for k in input.keys():
            input_dict[k].append(input[k])

    for k in input_dict.keys():
        input_dict[k] = torch.cat(input_dict[k])
    label_list = torch.cat(label_list)

    torch.save({'input':input_dict, 'label':label_list}, os.path.join(savedir,f'infer.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_config', type=str, help='config filename', default='exp.yaml')
    args = parser.parse_args()
    
    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)
    
    # DATA
    path = os.path.join(cfg['savedir'], 'infer.txt')
    file_list_infer = open(path, "r").read().split("\n")
    print("inference file 개수 : ",len(file_list_infer))

    # MODEL
    tokenizer = BertTokenizer.from_pretrained('KRBERT/vocab.txt', do_lower_case=False)
    model = BertForMaskedLM.from_pretrained("model_output/checkpoint-74000/")
    
    # MASK
    input_list = []
    label_list = []
    start = time()
    for i, file_path in enumerate(file_list_infer[:100]):
        masked_input, label = run(file_path, cfg, model, tokenizer, span_size=4)
        input_list.append(masked_input)
        label_list.append(label.input_ids)
    end = time()
    print('TIME :', end - start)

    # SAVE
    save(input_list, label_list, cfg["savedir"])


    
