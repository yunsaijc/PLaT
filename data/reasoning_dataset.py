# -*- coding: utf-8 -*-
"""
PLaT

：
1. ReasoningDataset: 
2. PLaTDataset: tokenization
3. StepGroupedBatchSampler: 
4. PLaTDataLoader: 
"""

import torch
from torch.utils.data import Dataset, Sampler, DataLoader
import json
import os
from typing import List, Dict, Any, Optional
from collections import defaultdict
import random
from datasets import Dataset as HFDataset, DatasetDict, load_dataset

from plat.config.base_config import (
    DataConfig, PLaTConfig,
    STEP_START, STEP_END, ANSWER_START, ANSWER_END
)


class ReasoningDataset(Dataset):
    """
    ： question||steps #### answer 
    """
    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at: {file_path}")
            
        self.samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parts = line.split('||')
                    question = parts[0]
                    solution_part = parts[1]
                    
                    solution_parts = solution_part.split('####')
                    solution = solution_parts[0].strip()
                    answer = solution_parts[1].strip()
                    
                    steps = [s.replace('<<', '').strip() for s in solution.split('>>') if s]
                    
                    self.samples.append({
                        'question': question,
                        'steps': steps,
                        'answer': answer,
                        'num_steps': len(steps) + 1  # +1 for answer
                    })
                except Exception as e:
                    print(f"Error parsing line: {line}. Error: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class PLaTDataset(Dataset):
    """
    PLaT（tokenization）
    """
    def __init__(self, raw_dataset, tokenizer, max_length: int = 128, target_max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_max_length = target_max_length
        
        if self.tokenizer.pad_token is None:
            raise ValueError("tokenizer.pad_token ！ tokenizer  pad token。")
        
        self.samples = []
        
        print("Preparing PLaT dataset...")
        
        for example in raw_dataset:
            question = example['question']
            steps = example['steps']
            answer = example['answer']
            
            sample = {
                'question': question,
                'steps': steps,
                'answer': answer,
                'num_steps': len(steps) + 1  # +1 for answer
            }
            self.samples.append(sample)
        
        total_steps = sum(s['num_steps'] for s in self.samples)
        print(f"Prepared {len(self.samples)} samples with {total_steps} total target steps")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        question = sample['question']
        steps = sample['steps']
        answer = sample['answer']
        num_total_steps = sample['num_steps']
        
        # Tokenize
        question_enc = self.tokenizer(
            question,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            padding_side='left',
            return_tensors='pt',
        )
        
        # （ + ）
        targets_input_ids = []
        targets_attention_mask = []
        
        # 
        for step in steps:
            target = f"{STEP_START}{step}{STEP_END}"
            
            target_enc = self.tokenizer(
                target,
                add_special_tokens=False,
                max_length=self.target_max_length,
                truncation=True,
                padding='max_length',
                padding_side='right',
                return_tensors='pt'
            )
            
            # pad_token_id-100
            target_ids = target_enc['input_ids'].squeeze(0).clone()
            target_ids[target_ids == self.tokenizer.pad_token_id] = -100
            
            targets_input_ids.append(target_ids)
            targets_attention_mask.append(target_enc['attention_mask'].squeeze(0))
        
        # 
        answer_target = f"{ANSWER_START}{answer}{ANSWER_END}"
        
        answer_target_enc = self.tokenizer(
            answer_target,
            add_special_tokens=False,
            max_length=self.target_max_length,
            truncation=True,
            padding='max_length',
            padding_side='right',
            return_tensors='pt'
        )
        
        answer_target_ids = answer_target_enc['input_ids'].squeeze(0).clone()
        answer_target_ids[answer_target_ids == self.tokenizer.pad_token_id] = -100
        
        targets_input_ids.append(answer_target_ids)
        targets_attention_mask.append(answer_target_enc['attention_mask'].squeeze(0))
        
        return {
            'question_input_ids': question_enc['input_ids'].squeeze(0),
            'question_attention_mask': question_enc['attention_mask'].squeeze(0),
            'targets_input_ids': targets_input_ids,
            'targets_attention_mask': targets_attention_mask,
            'num_steps': num_total_steps
        }


class StepGroupedBatchSampler(Sampler):
    """
    BatchSampler：batch
    
    collatepadding，batch。
    """
    def __init__(self, dataset, batch_size: int, shuffle: bool = True, 
                 drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.current_epoch = 0
        
        # 
        self.num_steps_to_indices = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            self.num_steps_to_indices[sample['num_steps']].append(idx)
        
        self.all_num_steps = sorted(self.num_steps_to_indices.keys())
        
        print(f"StepGroupedBatchSampler：")
        print(f"  : {len(dataset)}")
        print(f"  : {len(self.num_steps_to_indices)}")
        for num_steps, indices in sorted(self.num_steps_to_indices.items()):
            print(f"    {num_steps}: {len(indices)}")
    
    def set_epoch(self, epoch: int):
        """，epoch"""
        self.current_epoch = epoch
    
    def __iter__(self):
        # batches
        all_batches = []
        
        for num_steps, indices in self.num_steps_to_indices.items():
            # batch size：batch
            if num_steps >= 7:
                effective_batch_size = max(1, self.batch_size // 2)
            else:
                effective_batch_size = self.batch_size
            
            # 
            if self.shuffle:
                indices = indices.copy()
                random.shuffle(indices)
            
            # batch
            for i in range(0, len(indices), effective_batch_size):
                batch = indices[i:i+effective_batch_size]
                if len(batch) == effective_batch_size or not self.drop_last:
                    all_batches.append(batch)
        
        # batches
        if self.shuffle:
            random.shuffle(all_batches)
        
        for batch in all_batches:
            yield batch
    
    def __len__(self):
        total = 0
        for num_steps, indices in self.num_steps_to_indices.items():
            if num_steps >= 7:
                effective_batch_size = max(1, self.batch_size // 2)
            else:
                effective_batch_size = self.batch_size
            
            n = len(indices)
            if self.drop_last:
                total += n // effective_batch_size
            else:
                total += (n + effective_batch_size - 1) // effective_batch_size
        return total


def plat_collate_fn(batch):
    """
    PLaTcollate
    """
    # 
    num_steps = batch[0]['num_steps']
    assert all(item['num_steps'] == num_steps for item in batch), \
        "Batch！StepGroupedBatchSampler。"
    
    # 
    batch_question_input_ids = torch.stack([item['question_input_ids'] for item in batch])
    batch_question_attention_mask = torch.stack([item['question_attention_mask'] for item in batch])
    
    # targets
    batch_targets_input_ids = []
    batch_targets_attention_mask = []
    for tgt_idx in range(num_steps):
        step_targets_ids = [item['targets_input_ids'][tgt_idx] for item in batch]
        step_targets_mask = [item['targets_attention_mask'][tgt_idx] for item in batch]
        batch_targets_input_ids.append(torch.stack(step_targets_ids))
        batch_targets_attention_mask.append(torch.stack(step_targets_mask))
    
    return {
        'question_input_ids': batch_question_input_ids,
        'question_attention_mask': batch_question_attention_mask,
        'targets_input_ids': batch_targets_input_ids,
        'targets_attention_mask': batch_targets_attention_mask,
    }


def get_gsm8k_dataset(data_dir: str, cache_dir: str):
    """
    GSM8K
    
    ：
    1.  data_dir  JSON （gsm8k_train.json ）
    2.  cache_dir  JSON 
    3.  data_dir  .txt 
    
    Args:
        data_dir: （ JSON  .txt ）
        cache_dir: 
        
    Returns:
        DatasetDict:  train, valid, test 
    """
    
    os.makedirs(cache_dir, exist_ok=True)
    
    splits = ['train', 'valid', 'test']
    
    # 1:  data_dir  JSON 
    data_dir_json_files = {split: os.path.join(data_dir, f'gsm8k_{split}.json') for split in splits}
    if all(os.path.exists(f) for f in data_dir_json_files.values()):
        print("--- Loading dataset from preprocessed JSON files in data_dir... ---")
        datasets = {split: HFDataset.from_json(data_dir_json_files[split]) for split in splits}
        return DatasetDict(datasets)
    
    # 2:  cache_dir 
    cached_files = {split: os.path.join(cache_dir, f'gsm8k_{split}.json') for split in splits}
    if all(os.path.exists(f) for f in cached_files.values()):
        print("--- Loading dataset from cached JSON files... ---")
        datasets = {split: HFDataset.from_json(cached_files[split]) for split in splits}
        return DatasetDict(datasets)
    
    # 3:  .txt （）
    print("--- Processing from raw .txt files... ---")
    datasets = {}
    for split in splits:
        # 
        possible_paths = [
            os.path.join(data_dir, f'gsm_{split}.txt'),
            os.path.join(data_dir, 'gsm8k', f'gsm_{split}.txt'),
            os.path.join(data_dir, f'gsm8k_{split}.txt'),
        ]
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break
        
        if file_path is None:
            raise FileNotFoundError(
                f" {split} 。：{possible_paths}\n"
                f" data_dir  JSON （gsm8k_{split}.json） .txt "
            )
        
        raw_ds = ReasoningDataset(file_path)
        data_list = [sample for sample in raw_ds.samples]
        
        # 
        with open(cached_files[split], 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=4)
        print(f"--- Saved {split} data to cache: {cached_files[split]} ---")
        
        datasets[split] = HFDataset.from_list(data_list)
    
    return DatasetDict(datasets)


def load_svamp_dataset(data_dir: str, split: str = 'test'):
    """
    SVAMP
    
    Args:
        data_dir: 
        split: （train/test，traintest）
    
    Returns:
        HFDataset，question, answer
    """
    
    # SVAMPtrain+test
    train_file = os.path.join(data_dir, 'train.json')
    test_file = os.path.join(data_dir, 'test.json')
    
    data = []
    # train
    if os.path.exists(train_file):
        with open(train_file, 'r', encoding='utf-8') as f:
            data.extend(json.load(f))
    # test
    if os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8') as f:
            data.extend(json.load(f))
    
    # ：question, answer
    formatted_data = []
    for item in data:
        # SVAMP: Body + Question -> question, Answer -> answer
        question = f"{item['Body']} {item['Question']}".strip()
        answer = str(item['Answer'])  # 
        formatted_data.append({
            'question': question,
            'answer': answer
        })
    
    return HFDataset.from_list(formatted_data)


def load_multiarith_dataset(data_dir: str, split: str = 'test'):
    """
    MultiArith
    
    Args:
        data_dir: 
        split: （train/test）
    
    Returns:
        HFDataset，question, answer
    """
    
    json_file = os.path.join(data_dir, f'{split}.json')
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # ：question, answer
    formatted_data = []
    for item in data:
        # MultiArith: question -> question, final_ans -> answer
        question = item['question'].strip()
        answer = item['final_ans'].strip()
        formatted_data.append({
            'question': question,
            'answer': answer
        })
    
    return HFDataset.from_list(formatted_data)


def load_gsm_hard_dataset(data_dir: str, split: str = 'train'):
    """
    GSM-Hard
    
    Args:
        data_dir: 
        split: （train，gsm-hardtrain）
    
    Returns:
        HFDataset，question, answer
    """
    
    # gsm-hardparquet
    parquet_file = os.path.join(data_dir, 'data', f'{split}-00000-of-00001.parquet')
    if not os.path.exists(parquet_file):
        raise FileNotFoundError(f"GSM-Hard: {parquet_file}")
    
    dataset = load_dataset('parquet', data_files=parquet_file, split='train')
    data = dataset['train']
    
    # ：question, answer
    formatted_data = []
    for item in data:
        # gsm-hard: instruction -> question, response -> answer
        question = item['instruction'].strip()
        answer = str(item['response']).strip()
        formatted_data.append({
            'question': question,
            'answer': answer
        })
    
    return HFDataset.from_list(formatted_data)


def load_dataset_by_name(dataset_name: str, data_dir: str = None, split: str = 'test'):
    """
    
    
    Args:
        dataset_name: （gsm8k, svamp, multiarith, gsm-hard）
        data_dir: 
        split: （train/test）
    
    Returns:
        HFDataset
    """
    if data_dir is None:
        data_dir = '/home/jc/workspace/inf/plat/datasets/data'
    
    if dataset_name == 'gsm8k':
        return get_gsm8k_dataset(
            data_dir=data_dir,
            cache_dir=os.path.join(data_dir, 'cache')
        )[split]
    elif dataset_name == 'svamp':
        dataset_dir = os.path.join(data_dir, 'SVAMP')
        return load_svamp_dataset(dataset_dir, split='test')
    elif dataset_name == 'multiarith':
        dataset_dir = os.path.join(data_dir, 'MultiArith')
        return load_multiarith_dataset(dataset_dir, split='test')
    elif dataset_name == 'gsm-hard':
        dataset_dir = os.path.join(data_dir, 'gsm-hard')
        return load_gsm_hard_dataset(dataset_dir, split='train')
    else:
        raise ValueError(f": {dataset_name}")
