# -*- coding: utf-8 -*-
"""
PLaT model utility functions
"""

import torch
import torch.nn.functional as F
from typing import Any, Tuple, List, Optional, Dict
import re
from transformers import DynamicCache

from plat.config.base_config import ENCODE_TOKEN


def validate_equation(step_text: str) -> Tuple[bool, bool]:
    """
    Validate equation correctness
    
    Args:
        step_text: step text, may contain equation
        
    Returns:
        (has_equation, is_correct): has equation, is correct
    """
    if "=" not in step_text:
        return False, False
    
    # Try to extract equation
    match = re.search(r'([^=]+?)\s*=\s*([^=]+)', step_text)
    if not match:
        return False, False
    
    left_expr = match.group(1).strip()
    right_expr = match.group(2).strip()
    
    # Check if contains digits
    if not any(c.isdigit() for c in left_expr) and not any(c.isdigit() for c in right_expr):
        return False, False
    
    def safe_eval(expr: str) -> Tuple[bool, float]:
        """Safe evaluation of math expression"""
        try:
            expr = expr.replace(' ', '')
            if not re.match(r'^[0-9+\-*/().\s]+$', expr):
                return False, 0.0
            result = eval(expr)
            return True, float(result)
        except:
            return False, 0.0
    
    left_success, left_result = safe_eval(left_expr)
    if not left_success:
        return True, False
    
    right_success, right_result = safe_eval(right_expr)
    if not right_success:
        try:
            right_result = float(right_expr)
            right_success = True
        except:
            return True, False
    
    is_correct = abs(left_result - right_result) < 1e-6
    return True, is_correct


def extract_answer_from_text(text: str) -> str:
    """
    Extract final answer from generated text
    
    Args:
        text: generated text
        
    Returns:
        extracted_answer: extracted answer string
    """
    normalized_text = text.strip()
    
    # 1. Try to extract answer between markers
    pattern_special = r'<ANSWER_START>(.*?)<ANSWER_END>'
    match_special = re.search(pattern_special, normalized_text)
    if match_special:
        return match_special.group(1).strip()

    # 2. If no marker, extract last number
    numbers = re.findall(r'-?\d+\.?\d*', normalized_text)
    if numbers:
        return numbers[-1]
    
    return normalized_text


def extract_steps_from_text(text: str) -> List[str]:
    """
    Extract reasoning steps from generated text
    
    Args:
        text: generated text
        
    Returns:
        steps: extracted steps list
    """
    steps_special = re.findall(r'<STEP_START>(.*?)<STEP_END>', text)
    if steps_special:
        return [step.strip() for step in steps_special]
    
    return []


def compare_answers(pred: str, target: str) -> bool:
    """
    Compare two answers (numeric comparison)
    
    Args:
        pred: predicted answer
        target: true answer
        
    Returns:
        is_correct: is correct
    """
    if pred is None: return False
    pred_clean = re.sub(r'[^\d.-]', '', pred)
    target_clean = re.sub(r'[^\d.-]', '', target)
    
    try:
        pred_num = float(pred_clean)
        target_num = float(target_clean)
        return abs(pred_num - target_num) < 1e-6
    except (ValueError, TypeError):
        return pred.strip().lower() == target.strip().lower()


def batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move batch tensors to device"""
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def compute_metrics_from_states(states_pred: torch.Tensor, states_true: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics for state prediction
    
    Args:
        states_pred: predicted states
        states_true: true states
        
    Returns:
        metrics: metrics dictionary
    """
    with torch.no_grad():
        mse = F.mse_loss(states_pred, states_true).item()
        
        states_pred_flat = states_pred.reshape(-1, states_pred.size(-1))
        states_true_flat = states_true.reshape(-1, states_true.size(-1))
        
        cos_sim = F.cosine_similarity(states_pred_flat, states_true_flat, dim=-1).mean().item()
        l1_dist = F.l1_loss(states_pred, states_true).item()
        
    return {
        'state_mse': mse,
        'state_cosine_similarity': cos_sim,
        'state_l1_distance': l1_dist
    }


def format_reasoning_chain(steps: List[str], answer: str) -> str:
    """Format reasoning chain"""
    formatted_steps = '\n'.join([f"Step {i+1}: {step}" for i, step in enumerate(steps)])
    formatted_text = f"{formatted_steps}\nFinal Answer: {answer}"
    return formatted_text


def create_context_and_target(question: str, steps: List[str], step_idx: int,
                              step_start: str, step_end: str) -> tuple:
    """
    Create context and target for a step
    
    Args:
        question: question text
        steps: full steps list
        step_idx: current step index
        step_start: step start marker
        step_end: step end marker
        
    Returns:
        context: context text
        target: target text
    """
    if step_idx == 0:
        context = question
    else:
        previous_steps = ''.join([f"{step_start}{s}{step_end}" for s in steps[:step_idx]])
        context = question + previous_steps
    
    context += ENCODE_TOKEN
    target = f"{step_start}{steps[step_idx]}{step_end}"
    
    return context, target


def safe_slice(tensor, start, end):
    """Safe slice, handle None input"""
    return tensor[start:end] if tensor is not None else None


def compute_ema_cumulative(states, alpha=0.7):
    """
    Compute EMA cumulative
    
    Args:
        states: (B, T+1, latent_dim)
        alpha: decay factor
    Returns:
        ema_states: (B, T+1, latent_dim)
    """
    batch_size, seq_len, latent_dim = states.shape
    
    ema_states = [states[:, 0, :]]
    
    for t in range(1, seq_len):
        s_ema_t = alpha * states[:, t, :] + (1 - alpha) * ema_states[-1]
        ema_states.append(s_ema_t)
    
    return torch.stack(ema_states, dim=1)


def truncate_cache(cache, target_length):
    """Truncate cache to target length"""    
    current_length = cache.get_seq_length()
    
    if current_length <= target_length:
        return cache
    
    new_cache = DynamicCache()
    
    for layer_idx in range(len(cache)):
        key, value = cache[layer_idx]
        new_key = key[:, :, :target_length, :].detach()
        new_value = value[:, :, :target_length, :].detach()
        new_cache.update(new_key, new_value, layer_idx=layer_idx)
    
    return new_cache


def deepcopy_cache(obj):
    """
    Recursive deep copy
    """
    if obj is None:
        return None
    
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    
    if isinstance(obj, tuple):
        return tuple(deepcopy_cache(item) for item in obj)
    
    if isinstance(obj, list):
        return [deepcopy_cache(item) for item in obj]
    
    if isinstance(obj, dict):
        return {key: deepcopy_cache(value) for key, value in obj.items()}
    
    if hasattr(obj, '__dict__'):
        import copy
        new_obj = copy.copy(obj)
        for key, value in obj.__dict__.items():
            if isinstance(value, (torch.Tensor, list, tuple, dict)):
                setattr(new_obj, key, deepcopy_cache(value))
        return new_obj
    
    return obj
