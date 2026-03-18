# -*- coding: utf-8 -*-
"""
test_sft.py - SFT

：
1. （gsm8k, svamp, multiarith, gsm-hard）
2. pass@k  greedy 
3. 

:
    python test_sft.py --model_path /path/to/model --dataset gsm8k
    python test_sft.py --model_path /path/to/model --dataset gsm8k --k 5 --do_sample
"""
import os
import sys
import json
import argparse
import re
import time
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from plat.config.base_config import (
    STEP_START, STEP_END, ANSWER_START, ANSWER_END
)
from plat.data.reasoning_dataset import load_dataset_by_name
from plat.utils.helpers import extract_answer_from_text, compare_answers
from plat.models.cot_model import CoTModel

logger = logging.getLogger(__name__)


def extract_answer_number(text):
    """
    
    （）：
    1. "<ANSWER_START>42<ANSWER_END>" - PLaT
    2. "#### 42" - GSM8K
    3.  - fallback
    """
    # 1.  <ANSWER_START>XXX<ANSWER_END> 
    answer_match = re.search(r'<ANSWER_START>\s*(-?\d+\.?\d*)\s*<ANSWER_END>', text)
    if answer_match:
        return answer_match.group(1).strip()
    
    # 2. Fallback: 
    text_clean = text.split('<|endoftext|>')[0]
    numbers = re.findall(r'-?\d+\.?\d*', text_clean)
    if numbers:
        return numbers[-1].strip()
    
    return None


def generate_answer_batch(model, tokenizer, questions, device, max_new_tokens=256, 
                         do_sample=False, temperature=0.7, k=1):
    """
    （ pass@k）
    
    pass@k ：（ batch）
    
    Args:
        model: 
        tokenizer: tokenizer
        questions: 
        device: 
        max_new_tokens: 
        do_sample: 
        temperature: 
        k: （pass@k）
        
    Returns:
        batch_generated_texts: List[List[str]]， batch， k 
    """
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    
    try:
        batch_size = len(questions)
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id
        
        # token ID（）
        answer_start_id = tokenizer.convert_tokens_to_ids(ANSWER_START)
        answer_end_id = tokenizer.convert_tokens_to_ids(ANSWER_END)
        
        # token：ANSWER_END > PAD > EOS
        #  ANSWER_END 
        stop_token_ids = [pad_token_id, eos_token_id]
        if answer_end_id is not None and answer_end_id != tokenizer.unk_token_id:
            stop_token_ids.append(answer_end_id)
        
        # 
        inputs = tokenizer(questions, return_tensors='pt', truncation=True, max_length=256, padding=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        #  pass@k： batch  batch_size * k
        if k > 1:
            expanded_input_ids = input_ids.repeat_interleave(k, dim=0)
            expanded_attention_mask = attention_mask.repeat_interleave(k, dim=0)
        else:
            expanded_input_ids = input_ids
            expanded_attention_mask = attention_mask
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=expanded_input_ids,
                attention_mask=expanded_attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
        
        # ：[batch_size * k, seq_len] -> [batch_size, k, seq_len]
        generated_ids = outputs.view(batch_size, k, -1)
        
        # 
        batch_generated_texts = []
        input_length = input_ids.shape[1]
        
        for i in range(batch_size):
            samples = []
            for j in range(k):
                # 
                output_seq = generated_ids[i, j]
                gen_ids = output_seq[input_length:].tolist()
                
                #  ANSWER_END、PAD  EOS 
                stop_idx = len(gen_ids)
                for idx, token_id in enumerate(gen_ids):
                    if token_id in stop_token_ids:
                        stop_idx = idx
                        break
                gen_ids = gen_ids[:stop_idx]
                
                text = tokenizer.decode(gen_ids, skip_special_tokens=False)
                samples.append(text)
            
            batch_generated_texts.append(samples)
        
        return batch_generated_texts
    
    finally:
        tokenizer.padding_side = original_padding_side


def evaluate_model(model, tokenizer, dataset, device, max_samples=None, batch_size=8,
                   max_new_tokens=256, do_sample=False, temperature=0.7, k=1):
    """
    （ pass@k）
    
    Args:
        model: 
        tokenizer: 
        dataset: 
        device: 
        max_samples: 
        batch_size: 
        max_new_tokens: token
        do_sample: 
        temperature: 
        k: pass@k  k 
        
    Returns:
        results: 
    """
    model.eval()
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    correct = 0
    total = 0
    results = []
    
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    
    inference_start_time = time.time()
    all_reasoning_token_counts = []
    
    for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        batch_data = dataset.select(range(start_idx, end_idx))
        
        questions = [example['question'] for example in batch_data]
        gold_answers = [example['answer'] for example in batch_data]
        
        try:
            # 
            batch_samples = generate_answer_batch(
                model, tokenizer, questions, device,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                k=k
            )
            
            # 
            for i, (samples, gold_answer) in enumerate(zip(batch_samples, gold_answers)):
                # pass@k: ，
                is_correct_any = False
                for sample_text in samples:
                    # ，（）
                    pred_answer = extract_answer_from_text(sample_text)
                    if compare_answers(pred_answer, gold_answer):
                        is_correct_any = True
                        break
                
                if is_correct_any:
                    correct += 1
                total += 1
        
                results.append({
                            'question': questions[i],
                    'gold_answer': gold_answer,
                            'generated': samples[0],
                            'all_samples': samples if k > 1 else None,
                            'is_correct': is_correct_any
                        })
                        
        except Exception as e:
            print(f"\n:  {batch_idx} : {e}")
            for i, example in enumerate(batch_data):
                total += 1
                results.append({
                    'question': example['question'],
                    'gold_answer': example['answer'],
                    'generated': f"ERROR: {e}",
                    'is_correct': False
                })
    
    inference_end_time = time.time()
    inference_duration = inference_end_time - inference_start_time
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'details': results,
        'inference_time': inference_duration,
        'samples_per_second': total / inference_duration if inference_duration > 0 else 0,
        'avg_time_per_sample': inference_duration / total if total > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(description="SFT")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="SFT"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "svamp", "multiarith", "gsm-hard"],
        help=""
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/jc/workspace/inf/plat/data/data_file",
        help=""
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="split（gsm8k）"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="（）"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default='/home/jc/workspace/inf/plat/scripts/test_results/sft_test_results.json',
        help=""
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="token"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help=""
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="（ --k ）"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help=""
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="pass@k  k （1greedy）"
    )
    args = parser.parse_args()
    
    #  k > 1 ，（32）
    if args.k > 1 and not args.do_sample:
        args.do_sample = True
        logger.info(f"Auto-enabled do_sample=True for pass@{args.k}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # tokenizer
    logger.info(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    #  CoTModel.from_pretrained() （ backbone. ）
    model = CoTModel.from_pretrained(args.model_path)
    model.to(device)
    
    # 
    logger.info(f"Loading {args.dataset} dataset from {args.data_dir}")
    test_dataset = load_dataset_by_name(args.dataset, data_dir=args.data_dir, split=args.split)
    logger.info(f"Dataset loaded: {len(test_dataset)} samples")
    
    # 
    logger.info(f"Evaluating ({'Pass@' + str(args.k) if args.k > 1 else 'Greedy'})...")
    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        dataset=test_dataset,
        device=device,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        k=args.k
    )
    
    # 
    logger.info(f"\n=== Results ===")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Samples: {results['total']}")
    if args.k > 1:
        logger.info(f"Pass@{args.k}: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    else:
        logger.info(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    logger.info(f"Inference time: {results['inference_time']:.2f}s ({results['samples_per_second']:.2f} samples/s, {results['avg_time_per_sample']*1000:.2f}ms/sample)")
    
    # 
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model_path': args.model_path,
            'dataset': args.dataset,
            'split': args.split,
            'mode': f'pass@{args.k}' if args.k > 1 else 'greedy',
            'accuracy': results['accuracy'],
            'correct': results['correct'],
            'total': results['total'],
            'inference_time': results['inference_time'],
            'samples_per_second': results['samples_per_second'],
            'avg_time_per_sample': results['avg_time_per_sample'],
            'details': results['details']
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
    main()
