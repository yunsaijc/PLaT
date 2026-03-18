#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PLaT RL 
 plug/train_dyn_decoder.py 
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import re
import json
import builtins
import importlib.util
import yaml
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    PretrainedConfig,
    Trainer
)
import torch.distributed as dist

# 
sys.path.insert(0, '/home/jc/workspace/inf')

from plat.config.base_config import (
    PLaTConfig,
    STEP_START, STEP_END, ANSWER_START, ANSWER_END,
    ENCODE_TOKEN, DECODE_TOKEN, PLANNER_TOKEN,
    SPECIAL_TOKENS,
    RLConfig
)
from plat.data.reasoning_dataset import (
    PLaTDataset, get_gsm8k_dataset, plat_collate_fn
)
from plat.models.plat_model import PLaTModel

# =================================================================================
# 
# =================================================================================

def extract_answer_from_text(text: str) -> Optional[str]:
    """"""
    if ANSWER_START not in text:
        return None
    
    text = text.split(ANSWER_START)[1]
    if ANSWER_END in text:
        text = text.split(ANSWER_END)[0]
    
    text = text.strip()
    return text if text else None


def compare_answers(pred: str, gold: str) -> bool:
    """"""
    def clean_answer(ans):
        ans = str(ans).lower().strip()
        ans = re.sub(r'[,\s]', '', ans)
        ans = ans.replace('$', '')
        if '/' in ans:
            try:
                num, den = ans.split('/')
                return float(num) / float(den)
            except:
                pass
        try:
            return float(ans)
        except:
            return ans
    
    pred_clean = clean_answer(pred)
    gold_clean = clean_answer(gold)
    
    if isinstance(pred_clean, float) and isinstance(gold_clean, float):
        return abs(pred_clean - gold_clean) < 1e-6
    
    return pred_clean == gold_clean


def validate_equation(step_text: str) -> Tuple[bool, bool]:
    """"""
    eq_pattern = r'([^=]+)=([^=]+)'
    matches = re.findall(eq_pattern, step_text)
    
    if not matches:
        return False, False
    
    for left, right in matches:
        left = left.strip()
        right = right.strip()
        left = re.sub(r'[\$£€¥%]', '', left)
        right = re.sub(r'[\$£€¥%]', '', right)
        
        try:
            left_val = eval(left, {"__builtins__": {}}, {})
            right_val = eval(right, {"__builtins__": {}}, {})
            
            if isinstance(left_val, (int, float)) and isinstance(right_val, (int, float)):
                return True, abs(left_val - right_val) < 1e-6
        except:
            continue
    
    return False, False


def extract_steps_from_text(text: str) -> List[str]:
    """"""
    steps = []
    if STEP_START in text and STEP_END in text:
        pattern = rf'{re.escape(STEP_START)}(.*?){re.escape(STEP_END)}'
        steps = re.findall(pattern, text, re.DOTALL)
    return steps


# =================================================================================
# PLaT RL Trainer
# =================================================================================

class PLaTRLTrainer(Trainer):
    def __init__(self, 
                 processing_class=None, 
                 num_generations=4, 
                 ref_model=None, 
                 beta=0.1,
                 plat_config=None,
                 rl_config=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.processing_class = processing_class
        self.num_generations = num_generations
        self.ref_model = ref_model
        self.beta = beta
        self.plat_config = plat_config
        self.rl_config = rl_config
        
        #  plat_config 
        self.train_cfg = plat_config.train
        self.model_cfg = plat_config.model
        
        #  logger
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized PLaTRLTrainer with num_generations={num_generations}, beta={beta}")
    
    def get_train_dataloader(self):
        """ dataloader："""
        def collate_fn(batch):
            return {
                'questions': [item['question'] for item in batch],
                'answer': [item['answer'] for item in batch]
            }
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            drop_last=True
        )
    
    def get_eval_dataloader(self, eval_dataset=None):
        """ dataloader："""
        eval_dataset = eval_dataset or self.eval_dataset
        def collate_fn(batch):
            return {
                'questions': [item['question'] for item in batch],
                'answer': [item['answer'] for item in batch]
            }
        
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=False
        )
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """， OOM """
        device = next(model.parameters()).device
        try:
            return self._train_rl(model, inputs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.logger.warning(f"GPU OOM at step {getattr(self.state, 'global_step', 'unknown')}, step")
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                zero_loss = torch.tensor(0.0, device=device, requires_grad=False)
                return zero_loss
            raise
    
    def _run_rollout(self, model, inputs):
        """
         Rollout （ plug ）
        """
        questions = inputs['questions']
        answers = inputs['answer']
        device = next(model.parameters()).device
        
        module = model.module if hasattr(model, 'module') else model
        pad_id = self.processing_class.pad_token_id
        bos_id = getattr(self.processing_class, 'bos_token_id', None)
        eos_id = self.processing_class.eos_token_id
        special_ids = {tid for tid in [pad_id, bos_id, eos_id] if tid is not None}
        
        # RL 
        max_steps = self.rl_config.max_steps if self.rl_config else 20
        temperature = self.rl_config.temperature if self.rl_config else 0.9
        do_sample = self.rl_config.do_sample if self.rl_config else True
        train_intermediate = self.rl_config.train_intermediate_steps if self.rl_config else True
        
        batch_size = len(questions)
        
        # ： num_generations 
        all_prompts = []
        for q in questions:
            all_prompts.extend([q] * self.num_generations)
        
        # Tokenize
        orig_padding_side = self.processing_class.padding_side
        self.processing_class.padding_side = 'left'
        q_inputs = self.processing_class(all_prompts, return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)
        self.processing_class.padding_side = orig_padding_side
        
        #  generate_and_decode_rollout  states  ids
        # force_eval_mode=True eval，dropout
        #  plug ：rollout  planner adapter， decoder 
        if hasattr(module, 'set_adapter_mode'):
            module.set_adapter_mode('planner')
        with torch.no_grad():
            rollout_results = module.generate_and_decode_rollout(
                question_input_ids=q_inputs['input_ids'],
                question_attention_mask=q_inputs['attention_mask'],
                max_steps=max_steps,
                temperature=temperature,
                do_sample=do_sample,
                force_eval_mode=False,
                train_intermediate_steps=train_intermediate
            )
        
        # 
        all_ids_batch = rollout_results[0]  # List[Tensor]
        all_states_batch = rollout_results[3]  # List[Tensor], each (num_states, latent_dim)
        all_segments_batch = rollout_results[4]  # List[List[int]]
        
        # 
        step_samples = []
        
        # 
        for i in range(batch_size):
            sample_ans = answers[i]
            
            for k in range(self.num_generations):
                flatten_idx = i * self.num_generations + k
                
                gen_ids = all_ids_batch[flatten_idx]
                all_states = all_states_batch[flatten_idx].to(device)
                segment_lengths = all_segments_batch[flatten_idx]
                
                if all_states.size(0) == 0 or not segment_lengths:
                    continue
                
                num_latent = module.num_latent_for_a_step if hasattr(module, 'num_latent_for_a_step') else 1
                slot_accs = [all_states[j].clone() for j in range(min(num_latent, all_states.size(0)))]
                curr_token_idx = 0
                
                for step_idx, seg_len in enumerate(segment_lengths):
                    if curr_token_idx + seg_len > len(gen_ids):
                        break
                    
                    seg_tokens = gen_ids[curr_token_idx: curr_token_idx + seg_len]
                    seg_text = self.processing_class.decode(seg_tokens, skip_special_tokens=False).strip()
                    
                    is_step = seg_text.startswith(STEP_START) and seg_text.endswith(STEP_END)
                    is_answer = seg_text.startswith(ANSWER_START) and seg_text.endswith(ANSWER_END)
                    if not ((is_step and train_intermediate) or is_answer):
                        curr_token_idx += seg_len
                        continue
                    
                    state_chunks = torch.stack(slot_accs, dim=0).unsqueeze(0)
                    raw_ids = [tid.item() if hasattr(tid, 'item') else tid for tid in seg_tokens]
                    clean_ids = [tid for tid in raw_ids if tid not in special_ids]
                    if len(clean_ids) == 0:
                        curr_token_idx += seg_len
                        continue
                    
                    target_ids = torch.tensor([clean_ids], dtype=torch.long, device=device)
                    
                    step_samples.append({
                        'state_chunks': state_chunks,
                        'target_ids': target_ids,
                        'flatten_idx': flatten_idx,
                        'step_idx': step_idx,
                        'raw_ids': raw_ids,
                        'gen_text': seg_text,
                        'question': questions[i],
                        'answer': sample_ans
                    })
                    
                    next_s_idx = (step_idx + 1) * num_latent
                    if next_s_idx < all_states.size(0):
                        for slot_i in range(num_latent):
                            if next_s_idx + slot_i < all_states.size(0):
                                s_new = all_states[next_s_idx + slot_i]
                                if self.train_cfg.state_transition_mode == 'ema':
                                    alpha = self.train_cfg.ema_alpha
                                    slot_accs[slot_i] = alpha * s_new + (1 - alpha) * slot_accs[slot_i]
                                elif self.train_cfg.state_transition_mode == 'residual':
                                    slot_accs[slot_i] = slot_accs[slot_i] + s_new
                                elif self.train_cfg.state_transition_mode == 'independent':
                                    slot_accs[slot_i] = s_new
                    
                    curr_token_idx += seg_len
        
        #  old policy  log_prob（ PPO/GRPO ratio）
        if step_samples:
            if hasattr(module, 'set_adapter_mode'):
                module.set_adapter_mode('decoder')
            old_state_chunks = [s['state_chunks'] for s in step_samples]
            old_target_ids = [s['target_ids'] for s in step_samples]
            old_batch_chunks = torch.cat(old_state_chunks, dim=0)
            old_max_t_len = max(t.size(1) for t in old_target_ids)
            old_padded_ids = []
            for t in old_target_ids:
                ids = t[0].tolist() if t.size(0) > 0 else []
                old_padded_ids.append(ids + [pad_id] * (old_max_t_len - len(ids)))
            old_batch_targets = torch.tensor(old_padded_ids, device=device, dtype=torch.long)
            with torch.no_grad():
                old_logits, _ = module.decode_state(
                    state_chunks=old_batch_chunks,
                    target_ids=old_batch_targets,
                    decode_mode='train',
                    return_logits=True,
                    keep_logits_grad=True
                )
                old_logits = old_logits[:, :old_batch_targets.size(1), :]
                old_token_log_probs = -F.cross_entropy(
                    old_logits.reshape(-1, old_logits.size(-1)),
                    old_batch_targets.reshape(-1),
                    reduction='none',
                    ignore_index=pad_id
                ).reshape(old_batch_targets.size())
                old_sample_log_probs = old_token_log_probs.sum(dim=1)
            for i, sample in enumerate(step_samples):
                sample['old_log_prob'] = old_sample_log_probs[i].item()

        #  rollout 
        del all_states, slot_accs
        torch.cuda.empty_cache()
        
        return step_samples
    
    def reward_function(self, prompts, completions, **kwargs):
        """："""
        rewards = []
        gold_answers = kwargs.get('answer')
        
        train_intermediate = self.rl_config.train_intermediate_steps if self.rl_config else True
        
        # （ rl_config ）
        correct_answer_reward = self.rl_config.correct_answer_reward if self.rl_config else 1.0
        incorrect_answer_penalty = self.rl_config.incorrect_answer_penalty if self.rl_config else 0.0
        valid_answer_reward = self.rl_config.valid_answer_reward if self.rl_config else 0.1
        invalid_answer_penalty = self.rl_config.invalid_answer_penalty if self.rl_config else 0.1
        valid_eq_reward = self.rl_config.valid_equation_reward if self.rl_config else 0.1
        
        for idx, (pred, gold) in enumerate(zip(completions, gold_answers)):
            current_reward = 0.0
            
            is_step = not pred.startswith(ANSWER_START)
            is_answer = pred.startswith(ANSWER_START)
            
            # 1. 
            if is_step and train_intermediate:
                steps = extract_steps_from_text(pred)
                for step_text in steps:
                    has_equation, is_correct = validate_equation(step_text)
                    if has_equation:
                        current_reward += valid_eq_reward
                        if is_correct:
                            current_reward += valid_eq_reward
            
            # 2. 
            elif is_answer:
                pred_answer = extract_answer_from_text(pred)
                if pred_answer is not None:
                    for spec_token in [STEP_START, STEP_END, ENCODE_TOKEN, DECODE_TOKEN, PLANNER_TOKEN]:
                        if spec_token in pred_answer:
                            pred_answer = None
                            break
                
                if pred_answer is not None:
                    current_reward += valid_answer_reward
                    if compare_answers(pred_answer, gold):
                        current_reward += correct_answer_reward
                    else:
                        current_reward += incorrect_answer_penalty
                else:
                    current_reward += invalid_answer_penalty
            
            rewards.append(current_reward)
        
        return rewards
    
    def _train_rl(self, model, inputs):
        """RL ： PPO-like """
        step_samples = self._run_rollout(model, inputs)
        if not step_samples:
            return torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=False)
        
        device = next(model.parameters()).device
        pad_id = self.processing_class.pad_token_id
        
        # 
        batch_state_chunks = []
        batch_target_ids = []
        batch_raw_ids = []
        batch_questions = []
        batch_answers = []
        batch_gen_texts = []
        batch_old_log_probs = []
        batch_group_ids = []
        
        questions = inputs['questions']
        answers = inputs['answer']
        
        for sample in step_samples:
            batch_state_chunks.append(sample['state_chunks'])
            batch_target_ids.append(sample['target_ids'])
            batch_raw_ids.append(sample['raw_ids'])
            batch_questions.append(sample['question'])
            batch_answers.append(sample['answer'])
            batch_gen_texts.append(sample['gen_text'])
            # GRPO ：（i）
            batch_group_ids.append(sample['flatten_idx'] // self.num_generations)
            batch_old_log_probs.append(sample.get('old_log_prob'))
        
        if not batch_state_chunks:
            return torch.tensor(0.0, device=device, requires_grad=False)
        
        # 
        rewards_list = self.reward_function(
            prompts=batch_questions,
            completions=batch_gen_texts,
            answer=batch_answers
        )
        rewards = torch.tensor(rewards_list, device=device, dtype=torch.float32)
        
        #  log_probs
        module = model.module if hasattr(model, 'module') else model
        if hasattr(module, 'set_adapter_mode'):
            module.set_adapter_mode('decoder')

        batch_chunks = torch.cat(batch_state_chunks, dim=0)
        max_t_len = max(t.size(1) for t in batch_target_ids) if batch_target_ids else 1
        if max_t_len == 0:
            max_t_len = 1
        padded_ids = []
        for t in batch_target_ids:
            ids = t[0].tolist() if t.size(0) > 0 else []
            padded_ids.append(ids + [pad_id] * (max_t_len - len(ids)))
        batch_targets = torch.tensor(padded_ids, device=device, dtype=torch.long)
        
        #  decode_state  old/current policy  log_probs（ clip）
        old_logits = None
        old_token_log_probs = None
        use_cached_old_log_probs = all(v is not None for v in batch_old_log_probs)
        if batch_targets.size(1) > 0:
            if use_cached_old_log_probs:
                old_sample_log_probs = torch.tensor(batch_old_log_probs, device=device, dtype=torch.float32)
            else:
                model.eval()
                with torch.no_grad():
                    old_logits, _ = module.decode_state(
                        state_chunks=batch_chunks.detach(),
                        target_ids=batch_targets,
                        decode_mode='train',
                        return_logits=True,
                        keep_logits_grad=True
                    )
                    old_logits = old_logits[:, :batch_targets.size(1), :]
                    old_token_log_probs = -F.cross_entropy(
                        old_logits.reshape(-1, old_logits.size(-1)),
                        batch_targets.reshape(-1),
                        reduction='none',
                        ignore_index=pad_id
                    ).reshape(batch_targets.size())
                    old_sample_log_probs = old_token_log_probs.sum(dim=1)

            model.train()
            logits, _ = module.decode_state(
                state_chunks=batch_chunks,
                target_ids=batch_targets,
                decode_mode='train',
                return_logits=True,
                keep_logits_grad=True
            )
            
            logits = logits[:, :batch_targets.size(1), :]
            token_log_probs = -F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), 
                batch_targets.reshape(-1), 
                reduction='none',
                ignore_index=pad_id
            ).reshape(batch_targets.size())
            sample_log_probs = token_log_probs.sum(dim=1)
        else:
            # 
            old_sample_log_probs = torch.zeros(len(batch_state_chunks), device=device)
            sample_log_probs = torch.zeros(len(batch_state_chunks), device=device)
        
        # 
        should_log_details = (
            hasattr(self, 'state') and 
            hasattr(self.state, 'global_step') and
            self.state.global_step % self.args.logging_steps == 0
        )
        
        # KL Penalty
        if self.ref_model is not None and batch_targets.size(1) > 0:
            self.ref_model.eval()
            ref_device = next(self.ref_model.parameters()).device
            ref_dtype = next(self.ref_model.parameters()).dtype
            target_dtype = next(model.parameters()).dtype
            
            if ref_device != device:
                self.ref_model.to(device)
            if ref_dtype != target_dtype:
                self.ref_model.to(target_dtype)
            
            with torch.no_grad():
                ref_module = self.ref_model.module if hasattr(self.ref_model, 'module') else self.ref_model
                if hasattr(ref_module, 'set_adapter_mode'):
                    ref_module.set_adapter_mode('decoder')
                
                ref_logits, _ = ref_module.decode_state(
                    state_chunks=batch_chunks.detach(),
                    target_ids=batch_targets,
                    decode_mode='train',
                    return_logits=True,
                    keep_logits_grad=True
                )
                ref_logits = ref_logits[:, :batch_targets.size(1), :].to(device)
                ref_token_log_probs = -F.cross_entropy(
                    ref_logits.reshape(-1, ref_logits.size(-1)), 
                    batch_targets.reshape(-1), 
                    reduction='none',
                    ignore_index=pad_id
                ).reshape(batch_targets.size())
                ref_log_probs = ref_token_log_probs.sum(dim=1)
                
                kl = sample_log_probs.detach() - ref_log_probs.detach()
                rewards = rewards - self.beta * kl
        
        #  GRPO Advantage（ group ）
        group_ids = torch.tensor(batch_group_ids, device=device, dtype=torch.long)
        advantages = torch.zeros_like(rewards)
        for gid in torch.unique(group_ids):
            group_mask = (group_ids == gid)
            group_rewards = rewards[group_mask]
            if group_rewards.numel() <= 1:
                advantages[group_mask] = 0.0
            else:
                group_mean = group_rewards.mean()
                group_std = group_rewards.std(unbiased=False)
                advantages[group_mask] = (group_rewards - group_mean) / (group_std + 1e-8)
        
        # GRPO clipped objective
        clip_eps = getattr(self.rl_config, 'clip_eps', 0.2) if self.rl_config else 0.2
        log_ratio = (sample_log_probs - old_sample_log_probs).clamp(min=-10.0, max=10.0)
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

        # ：（ step ）
        # self.logger.info(f"  [MONITOR] sample_log_probs: mean={sample_log_probs.mean().item():.4f}, std={sample_log_probs.std().item():.4f}, min={sample_log_probs.min().item():.4f}, max={sample_log_probs.max().item():.4f}")
        # self.logger.info(f"  [MONITOR] old_sample_log_probs: mean={old_sample_log_probs.mean().item():.4f}, std={old_sample_log_probs.std().item():.4f}, min={old_sample_log_probs.min().item():.4f}, max={old_sample_log_probs.max().item():.4f}")
        # self.logger.info(f"  [MONITOR] ratio: mean={ratio.mean().item():.4f}, std={ratio.std().item():.4f}, min={ratio.min().item():.4f}, max={ratio.max().item():.4f}")
        # self.logger.info(f"  [MONITOR] advantages: mean={advantages.mean().item():.4f}, std={advantages.std().item():.4f}, min={advantages.min().item():.4f}, max={advantages.max().item():.4f}")
        # self.logger.info(f"  [MONITOR] rewards: mean={rewards.mean().item():.4f}, std={rewards.std().item():.4f}, min={rewards.min().item():.4f}, max={rewards.max().item():.4f}")

        pg_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # ： loss 
        loss_val_before_clip = (-(ratio * advantages).mean()).item()
        loss_clipped = (-(clipped_ratio * advantages).mean()).item()
        # self.logger.info(f"  [MONITOR] pg_loss (before clip): {loss_val_before_clip:.6f}, pg_loss (clipped): {loss_clipped:.6f}")
        # if abs(pg_loss.item()) > 1e6:
        #     self.logger.warning(f"  [WARN]  loss : {pg_loss.item():.6e}")

        if self.args.gradient_accumulation_steps > 1:
            pg_loss = pg_loss / self.args.gradient_accumulation_steps
        self.accelerator.backward(pg_loss)
        
        # 
        clip_frac = ((ratio - clipped_ratio).abs() > 1e-8).float().mean().item()
        if should_log_details:
            self.logger.debug(f"  RL Loss: {pg_loss.item():.4f}")
            self.logger.debug(f"  : {rewards.mean().item():.4f}, : {rewards.std().item():.4f}")
            self.logger.debug(f"  Clip : {clip_frac:.4f}, clip_eps={clip_eps}")
        
        self.last_reward_stats = {
            'reward_mean': rewards.mean().item(),
            'reward_std': rewards.std().item(),
            'advantage_mean': advantages.mean().item(),
            'advantage_std': advantages.std().item(),
            'ratio_mean': ratio.mean().item(),
            'ratio_std': ratio.std().item(),
            'sample_log_prob_mean': sample_log_probs.mean().item(),
            'old_sample_log_prob_mean': old_sample_log_probs.mean().item(),
            'clip_eps': float(clip_eps),
            'clip_frac': clip_frac,
            'num_samples': len(rewards)
        }

        loss_val = pg_loss.detach()
        has_target_tokens = batch_targets.size(1) > 0
        
        # 
        if has_target_tokens:
            del logits, token_log_probs
        del batch_chunks, batch_targets, old_sample_log_probs, sample_log_probs, ratio, clipped_ratio, pg_loss
        if old_logits is not None and old_token_log_probs is not None:
            del old_logits, old_token_log_probs
        if self.ref_model is not None and has_target_tokens:
            del ref_logits, ref_token_log_probs, ref_log_probs
        torch.cuda.empty_cache()
        
        return loss_val
    
    def compute_loss(self, model, inputs, num_items_in_batch=None):
        """ compute_loss"""
        return self.training_step(model, inputs, num_items_in_batch)
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """
         evaluation_loop， generate_and_decode_rollout 
        """
        import random
        import numpy as np
        from tqdm import tqdm
        
        # ，
        eval_seed = 42
        random.seed(eval_seed)
        np.random.seed(eval_seed)
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_seed)
        
        self.model.eval()
        module = self.model.module if hasattr(self.model, 'module') else self.model
        module.set_adapter_mode('planner')
        
        total_correct = 0
        total_samples = 0
        
        device = next(self.model.parameters()).device
        max_steps = self.rl_config.max_steps if self.rl_config else 20
        temperature = self.rl_config.temperature if self.rl_config else 0.9
        
        self.logger.info(f"（ generate_and_decode_rollout）...")
        
        for batch in tqdm(dataloader, desc=description):
            #  batch  question  answer
            if isinstance(batch, dict):
                questions = batch.get('questions', None)
                answers = batch.get('answer', None)
            else:
                continue
            
            if questions is None or answers is None:
                continue
            
            # 
            if isinstance(questions, str):
                questions = [questions]
            if isinstance(answers, str):
                answers = [answers]
            
            with torch.no_grad():
                # Tokenize
                orig_padding_side = self.processing_class.padding_side
                self.processing_class.padding_side = 'left'
                q_inputs = self.processing_class(questions, return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)
                self.processing_class.padding_side = orig_padding_side
                
                # 
                rollout_results = module.generate_and_decode_rollout(
                    question_input_ids=q_inputs['input_ids'],
                    question_attention_mask=q_inputs['attention_mask'],
                    max_steps=max_steps,
                    temperature=temperature,
                    do_sample=False,  #  greedy
                    force_eval_mode=True,
                    train_intermediate_steps=False
                )
                
                # 
                all_ids_batch = rollout_results[0]
                
                for i, gen_ids in enumerate(all_ids_batch):
                    gen_text = self.processing_class.decode(gen_ids, skip_special_tokens=False)
                    pred_answer = extract_answer_from_text(gen_text)
                    
                    if pred_answer is not None:
                        if compare_answers(pred_answer, answers[i]):
                            total_correct += 1
                    
                    total_samples += 1
        
        # 
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        self.logger.info(f": {total_correct}/{total_samples} = {accuracy:.4f}")
        
        #  Trainer 
        metrics = {f'{metric_key_prefix}_acc': accuracy}
        return type('EvalOutput', (), {
            'metrics': metrics,
            'predictions': None,
            'label_ids': None,
            'num_samples': total_samples
        })()


# =================================================================================
# Main
# =================================================================================

def setup_logging(output_dir: str, is_main_process: bool):
    """"""
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        log_filename = os.path.join(output_dir, f"rl_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        logger = logging.getLogger(__name__)
        logger.info(f": {log_filename}")
        
        original_print = builtins.print
        def print_main(*args, **kwargs):
            if is_main_process:
                original_print(*args, **kwargs)
        builtins.print = print_main
        
        return logger
    else:
        root_logger = logging.getLogger()
        root_logger.handlers = []
        root_logger.addHandler(logging.NullHandler())
        return logging.getLogger(__name__)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PLaT RL Training")
    parser.add_argument("--model_path", type=str, required=True, help="")
    parser.add_argument("--config", type=str, default=None, help="RL  (YAML)")
    parser.add_argument("--resume_from", type=str, default=None, help=" checkpoint ")
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    
    # 
    parser.add_argument("--num_generations", type=int, default=None, help="")
    parser.add_argument("--kl_beta", type=float, default=None, help="KL ")
    parser.add_argument("--max_steps", type=int, default=None, help="")
    parser.add_argument("--temperature", type=float, default=None, help="")
    args = parser.parse_args()
    


    # 1.  PLaT Config
    #  YAML ， model_path 
    if args.config:
        #  YAML 
        plat_config = PLaTConfig.load(args.config)
    else:
        #  model_path 
        plat_config = PLaTConfig.from_pretrained(args.model_path)


    #  rank（ torchrun/deepspeed ）
    rank_env = os.environ.get("RANK")
    local_rank_env = os.environ.get("LOCAL_RANK")
    if rank_env is not None:
        rank = int(rank_env)
    elif args.local_rank != -1:
        rank = int(args.local_rank)
    elif local_rank_env is not None:
        rank = int(local_rank_env)
    else:
        rank = -1
    is_main_process = (rank == -1) or (rank == 0)
    
    # 
    resumed_metadata = None
    inferred_wandb_run_id = None
    if args.resume_from:
        # resume  run ， run
        resume_path = os.path.normpath(args.resume_from)
        if os.path.basename(resume_path).startswith("checkpoint-"):
            output_dir = os.path.dirname(resume_path)
        else:
            output_dir = resume_path
        run_name = os.path.basename(output_dir)
        
        # （ checkpoint  run ）
        metadata_candidates = [
            os.path.join(resume_path, "training_metadata.json"),
            os.path.join(output_dir, "training_metadata.json"),
        ]
        for metadata_path in metadata_candidates:
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    resumed_metadata = json.load(f)
                break
        
        #  run_name，
        if resumed_metadata and resumed_metadata.get("run_name"):
            run_name = resumed_metadata["run_name"]

        # metadata ， wandb  run_id（ resume）
        if not (resumed_metadata and resumed_metadata.get("wandb_run_id")):
            wandb_dirs = [
                os.path.join(os.getcwd(), "wandb"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wandb"),
            ]
            matched_runs = []
            for wandb_dir in wandb_dirs:
                if not os.path.isdir(wandb_dir):
                    continue
                for entry in os.listdir(wandb_dir):
                    if not entry.startswith("run-"):
                        continue
                    run_dir = os.path.join(wandb_dir, entry)
                    output_log = os.path.join(run_dir, "files", "output.log")
                    if not os.path.isfile(output_log):
                        continue
                    try:
                        with open(output_log, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        if f"run_name={run_name}," in content:
                            run_id = entry.rsplit("-", 1)[-1]
                            matched_runs.append((os.path.getmtime(run_dir), run_id))
                    except Exception:
                        continue
            if matched_runs:
                #  run_id
                matched_runs.sort(key=lambda x: x[0], reverse=True)
                inferred_wandb_run_id = matched_runs[0][1]
    else:
        # ， run_name，
        sync_key = os.environ.get("TORCHELASTIC_RUN_ID", "default")
        launch_start_ts = datetime.now().timestamp()
        run_name_sync_file = os.path.join(
            plat_config.train.model_save_dir,
            f".rl_run_name_{sync_key}.txt"
        )
        if is_main_process:
            run_name = f"rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(plat_config.train.model_save_dir, exist_ok=True)
            with open(run_name_sync_file, "w") as f:
                f.write(run_name)
        else:
            wait_sec = 30
            run_name = None
            for _ in range(wait_sec * 10):
                if os.path.exists(run_name_sync_file):
                    # ，
                    if os.path.getmtime(run_name_sync_file) < launch_start_ts - 1.0:
                        import time
                        time.sleep(0.1)
                        continue
                    with open(run_name_sync_file, "r") as f:
                        candidate = f.read().strip()
                    if candidate:
                        run_name = candidate
                        break
                import time
                time.sleep(0.1)
            if run_name is None:
                raise RuntimeError(f" run_name: {run_name_sync_file}")
        output_dir = os.path.join(plat_config.train.model_save_dir, run_name)
    
    # 
    logger = setup_logging(output_dir, is_main_process)
    
    logger.info("=" * 60)
    logger.info("PLaT RL Training")
    logger.info("=" * 60)

    def persist_training_metadata(wandb_run=None):
        """， wandb run_id  resume"""
        if not is_main_process:
            return
        metadata_path = os.path.join(output_dir, "training_metadata.json")
        metadata_to_save = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    metadata_to_save = json.load(f)
            except Exception:
                metadata_to_save = {}
        metadata_to_save.update({
            "run_name": run_name,
            "wandb_run_id": wandb_run.id if wandb_run is not None else metadata_to_save.get("wandb_run_id"),
            "wandb_run_name": wandb_run.name if wandb_run is not None else metadata_to_save.get("wandb_run_name"),
        })
        with open(metadata_path, "w") as f:
            json.dump(metadata_to_save, f, indent=2)
    
    # Wandb
    if plat_config.wandb.enabled and is_main_process:
        import wandb
        os.environ["WANDB_PROJECT"] = plat_config.wandb.project
        resume_wandb_run_id = None
        if resumed_metadata and resumed_metadata.get("wandb_run_id"):
            resume_wandb_run_id = resumed_metadata["wandb_run_id"]
        elif inferred_wandb_run_id is not None:
            resume_wandb_run_id = inferred_wandb_run_id

        if resume_wandb_run_id:
            wandb.init(
                project=plat_config.wandb.project,
                name=run_name,
                id=resume_wandb_run_id,
                resume="must",
                config=plat_config.to_dict()
            )
        else:
            wandb.init(
                project=plat_config.wandb.project,
                name=run_name,
                config=plat_config.to_dict()
            )
        
        #  metadata， run_id 
        persist_training_metadata(wandb.run)
    elif plat_config.wandb.enabled and not is_main_process:
        os.environ["WANDB_MODE"] = "disabled"
    
    
    # 2.  Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    #  token
    for token in SPECIAL_TOKENS['additional_special_tokens']:
        if tokenizer.convert_tokens_to_ids(token) == tokenizer.unk_token_id:
            logger.warning(f" token: {token}, ...")
            tokenizer.add_special_tokens({'additional_special_tokens': [token]})
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f" pad_token: {tokenizer.pad_token}")
    
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # 3. 
    logger.info("...")
    print(f"data_dir: {plat_config.data}")
    print(f"data_dir: {plat_config.data.data_dir}")
    data_dir = plat_config.data.data_dir
    cache_dir = os.path.join(data_dir, "cache")
    raw_dataset = get_gsm8k_dataset(data_dir, cache_dir)
    
    if plat_config.data.debug_mode:
        logger.info(f"DEBUG MODE: ")
        raw_dataset['train'] = raw_dataset['train'].select(range(min(plat_config.data.debug_train_samples, len(raw_dataset['train']))))
    
    train_dataset = raw_dataset['train']
    logger.info(f": {len(train_dataset)}")
    
    # （ plug ： test， valid）
    eval_dataset = None
    if 'test' in raw_dataset:
        eval_dataset = raw_dataset['test']
        if plat_config.data.debug_mode:
            eval_dataset = eval_dataset.select(range(min(plat_config.data.debug_eval_samples, len(eval_dataset))))
        logger.info(f"(test): {len(eval_dataset)}")
    elif 'valid' in raw_dataset:
        eval_dataset = raw_dataset['valid']
        if plat_config.data.debug_mode:
            eval_dataset = eval_dataset.select(range(min(plat_config.data.debug_eval_samples, len(eval_dataset))))
        logger.info(f"(valid): {len(eval_dataset)}")
    
    # 4. 
    #  model_path （PLaT ， plat_config.yaml）
    logger.info(f" PLaT : {args.model_path}")
    model = PLaTModel.from_checkpoint(args.model_path)
    model.set_tokenizer(tokenizer)
    
    # 5.  RL （Dual LoRA）
    logger.info(" RL ...")
    if hasattr(model, 'prepare_for_rl_with_dual_lora'):
        model.prepare_for_rl_with_dual_lora()
    else:
        logger.warning(" Dual LoRA，")
    
    # ： LoRA 
    module = model.module if hasattr(model, 'module') else model
    dual_lora_enabled = getattr(module, 'dual_lora_enabled', False)
    has_peft = hasattr(module.backbone, 'peft_config')
    # logger.info(f"  [DIAG] dual_lora_enabled={dual_lora_enabled}, has_peft_config={has_peft}")
    if has_peft:
        pass
        # logger.info(f"  [DIAG] peft_config keys: {list(module.backbone.peft_config.keys())}")
    
    # 
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f": {total_params:,} | : {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # 6. （ KL ）
    ref_model = None
    logger.info("...")
    ref_model = PLaTModel.from_checkpoint(args.model_path)
    ref_model.set_tokenizer(tokenizer)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    logger.info("")
    
    # 7. RL （ YAML  plat_config ，）
    #  plat_config  RL 
    rl_config = plat_config.rl
    
    # 
    if args.num_generations is not None:
        rl_config.num_generations = args.num_generations
    if args.kl_beta is not None:
        rl_config.kl_beta = args.kl_beta
    if args.max_steps is not None:
        rl_config.max_steps = args.max_steps
    if args.temperature is not None:
        rl_config.temperature = args.temperature
    
    logger.info(f"RL : {rl_config}")
    
    # 8. 
    #  wandb  report_to
    report_to = "wandb" if plat_config.wandb.enabled else "none"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        report_to=report_to,
        num_train_epochs=plat_config.train.num_train_epochs,
        per_device_train_batch_size=plat_config.train.per_device_train_batch_size,
        per_device_eval_batch_size=plat_config.train.per_device_eval_batch_size,
        gradient_accumulation_steps=plat_config.train.gradient_accumulation_steps,
        learning_rate=plat_config.train.learning_rate,
        warmup_ratio=plat_config.train.warmup_ratio,
        lr_scheduler_type=plat_config.train.lr_scheduler_type,
        weight_decay=plat_config.train.weight_decay,
        logging_steps=plat_config.train.logging_steps,
        eval_strategy=plat_config.train.eval_strategy,
        eval_steps=plat_config.train.eval_steps,
        save_strategy=plat_config.train.save_strategy,
        save_steps=plat_config.train.save_steps,
        save_total_limit=plat_config.train.save_total_limit,
        load_best_model_at_end=plat_config.train.load_best_model_at_end,
        metric_for_best_model=plat_config.train.metric_for_best_model,
        bf16=plat_config.train.bf16,
        fp16=plat_config.train.fp16,
        deepspeed=plat_config.train.deepspeed,
        remove_unused_columns=False,
        dataloader_num_workers=plat_config.train.dataloader_num_workers,
        seed=plat_config.train.seed,
        resume_from_checkpoint=args.resume_from,
    )
    
    logger.info(f"Training arguments: {training_args}")
    
    # 9.  Trainer
    trainer = PLaTRLTrainer(
        processing_class=tokenizer,
        num_generations=rl_config.num_generations,
        ref_model=ref_model,
        beta=rl_config.kl_beta,
        plat_config=plat_config,
        rl_config=rl_config,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # 10. 
    logger.info(" RL ...")
    trainer.train(resume_from_checkpoint=args.resume_from if args.resume_from else None)

    #  metadata，
    if plat_config.wandb.enabled and is_main_process:
        import wandb
        persist_training_metadata(wandb.run)
    
    # 11. 
    final_output_dir = os.path.join(output_dir, "final")
    logger.info(f": {final_output_dir}")
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    logger.info("!")
    
    #  Wandb
    if plat_config.wandb.enabled and is_main_process:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
