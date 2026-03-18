#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_plat.py - PLaT

：
1. PLaT
2. 
3. 
4. GPU
5. lazypass@k
"""

import os
import sys
import argparse
import json
import torch
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Any
import random
import numpy as np
import time
import torch.multiprocessing as mp

# 
sys.path.insert(0, '/home/jc/workspace/inf')

from transformers import AutoTokenizer
from plat.config.base_config import PLaTConfig, SPECIAL_TOKENS
from plat.data.reasoning_dataset import get_gsm8k_dataset
from plat.models.plat_model import PLaTModel
from plat.utils.helpers import compare_answers, extract_answer_from_text


def single_worker_process(worker_id, gpu_id, model_path, data_samples, args, return_dict, progress_counter):
    """worker，workerGPU"""
    try:
        # 
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # GPU
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
        
        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 
        model = PLaTModel.from_checkpoint(model_path)
        model.set_tokenizer(tokenizer)
        
        # embeddingtokenizer
        target_vocab_size = len(tokenizer)
        current_vocab_size = model.backbone.get_input_embeddings().weight.size(0)
        if current_vocab_size < target_vocab_size:
            model.backbone.resize_token_embeddings(target_vocab_size, mean_resizing=False)
            model.backbone.config.vocab_size = target_vocab_size
            src_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            with torch.no_grad():
                emb = model.backbone.get_input_embeddings().weight
                for tid in range(current_vocab_size, target_vocab_size):
                    emb[tid].copy_(emb[src_id])
        
        model.to(device)
        model.eval()
        
        results = []
        
        for sample in data_samples:
            question = sample['question']
            target_answer = sample['answer']
            target_steps = sample.get('steps', [])
            
            try:
                if args.lazy:
                    final_answers, reasoning_steps, token_stats = model.inference_lazy(
                        question,
                        max_steps=args.max_steps,
                        pass_at_k=args.pass_at_k,
                        temperature=args.temperature,
                        do_sample=args.do_sample,
                        extract_answer=True
                    )
                else:
                    final_answers, reasoning_steps, token_stats = model.inference(
                        question,
                        max_steps=args.max_steps,
                        pass_at_k=args.pass_at_k,
                        temperature=args.temperature,
                        do_sample=args.do_sample,
                        extract_answer=True
                    )
                
                pred_answer = final_answers[0] if final_answers else ""
                pred_steps = reasoning_steps[0] if reasoning_steps else []
                
                # 
                is_correct = any(compare_answers(pa, target_answer) for pa in final_answers)
                
                result = {
                    'index': sample.get('original_idx', 0),
                    'question': question,
                    'target_answer': target_answer,
                    'target_steps': target_steps,
                    'pred_answer': pred_answer,
                    'pred_steps': pred_steps,
                    'all_pred_answers': final_answers,
                    'is_correct': is_correct,
                    'token_stats': token_stats,
                }
                
            except Exception as e:
                raise e
                result = {
                    'index': sample.get('original_idx', 0),
                    'question': question,
                    'target_answer': target_answer,
                    'pred_answer': "",
                    'is_correct': False,
                    'error': str(e)
                }
            
            results.append(result)
            
            # 
            with progress_counter.get_lock():
                progress_counter.value += 1
        
        return_dict[worker_id] = {
            'results': results,
            'num_samples': len(data_samples)
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return_dict[worker_id] = {
            'results': [],
            'num_samples': 0,
            'error': str(e)
        }


def distribute_data_to_workers(data, num_workers):
    """worker"""
    total = len(data)
    chunk_size = (total + num_workers - 1) // num_workers
    
    chunks = []
    for i in range(num_workers):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total)
        if start_idx < total:
            chunk = []
            for j in range(start_idx, end_idx):
                sample = data[j]
                sample['original_idx'] = j
                chunk.append(sample)
            chunks.append(chunk)
    
    return chunks


def evaluate_on_dataset_single(model, tokenizer, dataset, config, 
                               max_samples: int = None,
                               max_steps: int = 20,
                               decode_context_mode: str = 'seq',
                               temperature: float = 1.0,
                               do_sample: bool = False,
                               lazy: bool = False,
                               pass_at_k: int = 1,
                               verbose: bool = False) -> Dict[str, Any]:
    """（）"""
    model.eval()
    device = next(model.parameters()).device
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    results = []
    correct_count = 0
    total_count = 0
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    for idx, example in enumerate(tqdm(dataset)):
        question = example['question']
        target_answer = example['answer']
        target_steps = example.get('steps', [])
        
        try:
            if lazy:
                final_answers, reasoning_steps, token_stats = model.inference_lazy(
                    question,
                    max_steps=max_steps,
                    pass_at_k=pass_at_k,
                    temperature=temperature,
                    do_sample=do_sample,
                    extract_answer=True
                )
            else:
                final_answers, reasoning_steps, token_stats = model.inference(
                    question,
                    max_steps=max_steps,
                    pass_at_k=pass_at_k,
                    temperature=temperature,
                    do_sample=do_sample,
                    extract_answer=True
                )
            
            pred_answer = final_answers[0] if final_answers else ""
            pred_steps = reasoning_steps[0] if reasoning_steps else []
            
            # 
            is_correct = any(compare_answers(pa, target_answer) for pa in final_answers)
            
            if is_correct:
                correct_count += 1
            total_count += 1
            
            result = {
                'idx': idx,
                'question': question,
                'target_answer': target_answer,
                'pred_answer': pred_answer,
                'all_pred_answers': final_answers,
                'is_correct': is_correct,
                'target_steps': target_steps,
                'pred_steps': pred_steps,
                'token_stats': token_stats,
            }
            results.append(result)
            
            if verbose or (not is_correct and idx < 10):
                print(f"\n{'='*50}")
                print(f"[{idx}] Question: {question}")
                print(f"Target: {target_answer}")
                print(f"Pred: {pred_answer}")
                print(f"Correct: {is_correct}")
                if pred_steps:
                    print(f"Steps: {pred_steps[:3]}...")
                    
        except Exception as e:
            raise e
            total_count += 1
    
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'correct_count': correct_count,
        'total_count': total_count,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(description="PLaT（GPU）")
    parser.add_argument("--model_path", type=str, required=True, help="")
    parser.add_argument("--config", type=str, default=None, help="")
    parser.add_argument("--output_dir", type=str, default=None, help="")
    parser.add_argument("--data_dir", type=str, default="/home/jc/workspace/inf/coconut/data",
                        help="")
    parser.add_argument("--cache_dir", type=str, default="/home/jc/workspace/inf/plat/data/gsm8k_processed",
                        help="")
    parser.add_argument("--split", type=str, default="test", choices=['train', 'valid', 'test'],
                        help="")
    parser.add_argument("--max_samples", type=int, default=None, help="")
    parser.add_argument("--max_steps", type=int, default=20, help="")
    parser.add_argument("--decode_mode", type=str, default="seq", help="")
    parser.add_argument("--temperature", type=float, default=0.9, help="")
    parser.add_argument("--do_sample", action="store_true", default=False, help="")
    parser.add_argument("--pass_at_k", type=int, default=1, help="pass@k")
    parser.add_argument("--lazy", type=bool, default=True, help="lazy")
    parser.add_argument("--verbose", action="store_true", help="")
    parser.add_argument("--device", type=str, default="cuda", help="")
    
    # 
    parser.add_argument("--gpus", type=str, default='0,1', help="GPU，")
    parser.add_argument("--workers_per_gpu", type=int, default=4, help="GPU")
    parser.add_argument("--seed", type=int, default=None, help="")
    
    args = parser.parse_args()
    
    # 
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # GPU，
    use_parallel = args.workers_per_gpu > 1 and torch.cuda.is_available()
    
    print(f"args: {args}")
    
    # 
    print(f"Loading {args.split} dataset...")
    datasets = get_gsm8k_dataset(data_dir=args.data_dir, cache_dir=args.cache_dir)
    test_dataset = datasets[args.split]
    print(f"Dataset loaded: {len(test_dataset)} samples")
    
    if args.max_samples:
        test_dataset = test_dataset.select(range(min(args.max_samples, len(test_dataset))))
    
    if use_parallel:
        # 
        _run_parallel(test_dataset, args)
    else:
        # 
        _run_single(test_dataset, args)


def _run_single(test_dataset, args):
    """"""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 
    from plat.models.plat_model import PLaTModel
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = PLaTModel.from_checkpoint(args.model_path)
    model.set_tokenizer(tokenizer)
    
    # embeddingtokenizer
    target_vocab_size = len(tokenizer)
    current_vocab_size = model.backbone.get_input_embeddings().weight.size(0)
    if current_vocab_size < target_vocab_size:
        model.backbone.resize_token_embeddings(target_vocab_size, mean_resizing=False)
        model.backbone.config.vocab_size = target_vocab_size
        src_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        with torch.no_grad():
            emb = model.backbone.get_input_embeddings().weight
            for tid in range(current_vocab_size, target_vocab_size):
                emb[tid].copy_(emb[src_id])
    
    model = model.to(device)
    print(f"Model loaded. Device: {device}")
    
    config = model.plat_config
    
    # 
    results = evaluate_on_dataset_single(
        model=model,
        tokenizer=tokenizer,
        dataset=test_dataset,
        config=config,
        max_steps=args.max_steps,
        decode_context_mode=args.decode_mode,
        temperature=args.temperature,
        do_sample=args.do_sample,
        lazy=args.lazy,
        pass_at_k=args.pass_at_k,
        verbose=args.verbose
    )
    
    # 
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Correct: {results['correct_count']} / {results['total_count']}")
    
    # 
    _save_results(results, args)


def _run_parallel(test_dataset, args):
    """"""
    gpus = [int(g) for g in args.gpus.split(',')]
    num_gpus = len(gpus)
    total_workers = num_gpus * args.workers_per_gpu
    
    print(f"{'='*80}")
    print(f"Multi-GPU Multi-Process Inference Configuration")
    print(f"{'='*80}")
    print(f"Model Path: {args.model_path}")
    print(f"GPUs: {gpus}")
    print(f"Workers per GPU: {args.workers_per_gpu}")
    print(f"Total parallel workers: {total_workers}")
    print(f"Pass@k: {args.pass_at_k}")
    print(f"Temperature: {args.temperature}")
    print(f"Do Sample: {args.do_sample}")
    print(f"Lazy: {args.lazy}")
    print(f"Seed: {args.seed}")
    print(f"{'='*80}")
    
    # worker
    data_chunks = distribute_data_to_workers(test_dataset, total_workers)
    
    # 
    for gpu_idx, gpu_id in enumerate(gpus):
        worker_start = gpu_idx * args.workers_per_gpu
        worker_end = (gpu_idx + 1) * args.workers_per_gpu
        total_samples_on_gpu = sum(len(data_chunks[i]) for i in range(worker_start, min(worker_end, len(data_chunks))))
        print(f"  GPU {gpu_id}: {total_samples_on_gpu} samples")
    
    # 
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    progress_counter = mp.Value('i', 0)
    processes = []
    
    inference_start_time = time.time()
    
    for worker_id in range(len(data_chunks)):
        gpu_idx = worker_id // args.workers_per_gpu
        gpu_id = gpus[gpu_idx]
        p = mp.Process(
            target=single_worker_process,
            args=(worker_id, gpu_id, args.model_path, data_chunks[worker_id], 
                  args, return_dict, progress_counter)
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)
    
    # 
    with tqdm(total=len(test_dataset), desc="Processing", unit="sample") as pbar:
        last_value = 0
        while any(p.is_alive() for p in processes):
            current_value = progress_counter.value
            if current_value > last_value:
                pbar.update(current_value - last_value)
                last_value = current_value
            time.sleep(0.1)
        
        current_value = progress_counter.value
        if current_value > last_value:
            pbar.update(current_value - last_value)
    
    # 
    for p in processes:
        p.join()
    
    inference_end_time = time.time()
    inference_duration = inference_end_time - inference_start_time
    
    # 
    print("\nCollecting results from all workers...")
    all_results = []
    for worker_id in range(len(data_chunks)):
        if worker_id in return_dict:
            worker_data = return_dict[worker_id]
            if isinstance(worker_data, dict) and 'results' in worker_data:
                all_results.extend(worker_data['results'])
    
    # 
    all_results.sort(key=lambda x: x.get('index', 0))
    
    # 
    correct = sum(1 for r in all_results if r.get('is_correct', False))
    total = len(all_results)
    accuracy = correct / total if total > 0 else 0
    
    # token
    total_latent_states = 0
    total_tokens = 0
    token_stats_count = 0
    
    for r in all_results:
        if 'token_stats' in r and r['token_stats'] is not None:
            ts = r['token_stats']
            total_latent_states += ts.get('num_latent_states', 0)
            total_tokens += ts.get('total_tokens', 0)
            token_stats_count += 1
    
    avg_latent_states = total_latent_states / token_stats_count if token_stats_count > 0 else 0
    avg_tokens = total_tokens / token_stats_count if token_stats_count > 0 else 0
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Total samples processed: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Pass@{args.pass_at_k} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f": {inference_duration:.2f} | : {inference_duration/total:.3f}")
    print(f"Average latent states: {avg_latent_states:.2f}")
    print(f"Average tokens: {avg_tokens:.2f}")
    print("="*80)
    
    results = {
        'accuracy': accuracy,
        'correct_count': correct,
        'total_count': total,
        'results': all_results
    }
    
    _save_results(results, args)


def _save_results(results, args):
    """"""
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(args.model_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sample_mode = "sample" if args.do_sample else "greedy"
    output_file = os.path.join(output_dir, f"plat_test_{args.split}_{sample_mode}_{timestamp}.json")
    
    output_data = {
        'model_path': args.model_path,
        'split': args.split,
        'max_steps': args.max_steps,
        'pass_at_k': args.pass_at_k,
        'temperature': args.temperature,
        'do_sample': args.do_sample,
        'lazy': args.lazy,
        'accuracy': results['accuracy'],
        'correct_count': results['correct_count'],
        'total_count': results['total_count'],
        'results': results['results']
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
