from glob import glob
import os
import json
import numpy as np
import argparse
import yaml
import torch
import random
import time
from datetime import datetime

import wandb

from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertForMaskedLM, BertTokenizer, BertConfig
from transformers import Trainer, TrainingArguments

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)


# 2. 데이터 불러오기
class PaddedDataset(Dataset):
    def __init__(self, file_list, tokenizer, max_seq_length, PAD = False):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.file_list = file_list
        self.PAD = PAD
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        input = self._get_text(self.file_list[idx])
        source = self.tokenizer(input, max_length = self.max_seq_length, 
                                padding = "max_length", truncation = True, 
                                )
        source_ids, target_ids = self.mask_token(source['input_ids'])

        if 'Clickbait_Direct' in self.file_list[idx]:
            token_type_ids = torch.tensor([1] * self.max_seq_length, dtype = torch.long)
        else:
            token_type_ids = torch.tensor([0] * self.max_seq_length, dtype = torch.long)

        return {
            "input_ids" : torch.tensor(source_ids, dtype=torch.long), 
            "attention_mask" : torch.tensor(source['attention_mask'], dtype=torch.long), 
            "token_type_ids" : token_type_ids,
            "labels" : torch.tensor(target_ids, dtype = torch.long)
            }

    def _get_text(self, file_path):
        source_file = json.load(open(file_path, "r"))
        title = source_file['sourceDataInfo']['newsTitle']
        content = source_file['sourceDataInfo']['newsContent']
        input_text = title + '[SEP]' + content
        return input_text
    
    def mask_token(self, input_ids : list, n = 4): 
        """
        BertTokenizer [SEP]를 사용한다고 가정
        input : w1, ..., wi-1, [MASK][MASK][MASK][MASK], wi+3, 
        label : 
        """
        label = input_ids.copy()

        # 1. title 부분에 [MASK]처리하기
        input_ids = np.array(input_ids)
        content_idx = np.where(input_ids == self.tokenizer.sep_token_id)[0]

        if self.PAD == False:
            rand_idx = random.randint(1,content_idx[0]-n) #[CLS] w1, w2, ..., wk, [SEP]에서 [SEP]이 겹치지 않게 mask하기

            ## input [MASK]처리하기
            input_ids[rand_idx : rand_idx+n] = self.tokenizer.mask_token_id
            label[:rand_idx] = -100  # We only compute loss on masked tokens
            label[rand_idx+n:] = -100

        elif self.PAD == True :
            ## 실제 mask 토큰의 개수(k) 구하기(1~4)
            n_masked = random.randint(1, n)
            rand_idx = random.randint(1,content_idx[0]-n_masked) #[CLS] w1, w2, ..., wk, [SEP]에서 [SEP]이 겹치지 않게 mask하기

            ## input [MASK]처리하기
            input_ids[rand_idx : rand_idx+n_masked] = self.tokenizer.mask_token_id

            ## pad token추가 되는 부분까지 [MASK]추가하기
            if n_masked != n :
                input_ids = np.hstack((input_ids[:rand_idx+n_masked], np.full(n-n_masked, self.tokenizer.mask_token_id), 
                                    input_ids[rand_idx+n_masked:]))
            ## label에 [PAD] 추가하기   
            label = np.hstack((label[:rand_idx+n_masked], np.full(n-n_masked, self.tokenizer.pad_token_id),
                            label[rand_idx+n_masked:],))

            # 2. loss계산 안 할 부분 찾기 : special token(cls, sep) + content
            label[:rand_idx] = -100  # We only compute loss on masked tokens
            label[rand_idx+n:] = -100

            ## maxlen 맞추기
            input_ids = np.hstack((input_ids[:self.max_seq_length-1], [tokenizer.sep_token_id]))
            label = label[:self.max_seq_length]

        return input_ids, label

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_config', type=str, help='config filename', default='exp.yaml')
    args = parser.parse_args()
    
    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)

    # set seed
    torch_seed(cfg['SEED'])

    # Load file list
    ## nonbait데이터 training과 inference로 나누기
    nonbait_filelist = glob(os.path.join(cfg['datadir'], '[!sample]*/Clickbait_Auto/*/*'))
    train_size = cfg['DATASET']['num_train_nonbait']
    inference_size = len(nonbait_filelist) - train_size
    nonbait_train, nonbait_infer = random_split(dataset = nonbait_filelist, 
                                                lengths = [train_size, inference_size], 
                                                generator = torch.Generator().manual_seed(42)
                                                )
    nonbait_train_list = [nonbait_filelist[i] for i in nonbait_train.indices]
    nonbait_infer_list = [nonbait_filelist[i] for i in nonbait_infer.indices]

    bait_filelist = glob(os.path.join(cfg['datadir'], '[!sample]*/Clickbait_Direct/*/*'))
    file_list = nonbait_train_list + bait_filelist

    # Tokenizer 불러오기  
    tokenizer = BertTokenizer.from_pretrained('KRBERT/vocab.txt', do_lower_case=False)

    # Dataset
    train_size = int(0.8 * len(file_list))
    test_size = len(file_list) - train_size

    train_idx, test_idx = random_split(file_list, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    train_list=[file_list[i] for i in train_idx.indices]
    test_list=[file_list[i] for i in test_idx.indices]

    trainset = PaddedDataset(train_list, tokenizer, max_seq_length=512)
    testset = PaddedDataset(test_list, tokenizer, max_seq_length=512)

    # Model
    config = BertConfig(cfg["config"])
    model = BertForMaskedLM.from_pretrained(cfg['model'], config=cfg['config'])

    # Training
    current_datetime = str(datetime.now()) #디렉토리 덮어씌어지지 않게 현재 시간 포함해서 디렉토리 이름 설정
    output_dir = 'model_output' + current_datetime
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_gpu_train_batch_size=8,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100,
        report_to="wandb",
        run_name="PaddedMLM-1"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=trainset,
        eval_dataset=testset,
    )

    trainer.train()
    trainer.save_model('./model_output')