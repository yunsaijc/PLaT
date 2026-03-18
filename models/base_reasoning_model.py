# -*- coding: utf-8 -*-
"""
BaseReasoningModel - 

（PLaT、CoT），。
（PLaTModel、CoTModel）。
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
from typing import Optional, List, Dict, Any, Tuple
import logging
from dataclasses import dataclass
from transformers.utils import ModelOutput

try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from plat.config.base_config import (
    STEP_START, STEP_END, ANSWER_START, ANSWER_END,
    ENCODE_TOKEN, DECODE_TOKEN, PLANNER_TOKEN
)

logger = logging.getLogger(__name__)



@dataclass
class ReasoningOutput(ModelOutput):
    """"""
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None



class BaseReasoningModel(PreTrainedModel, ABC):
    """
    
    
    ：
    - set_tokenizer(): tokenizertoken ID
    - encode_state(): （）
    - forward(): 
    - inference(): 
    
    ：
    - _init_components(): 
    - forward(): 
    - inference(): 
    """
    config_class = PretrainedConfig
    
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        
        # Public attributes
        self.tokenizer = None
        self.encode_token_id = None
        self.decode_token_id = None
        self.planner_token_id = None
        self.step_start_id = None
        self.step_end_id = None
        self.answer_start_id = None
        self.answer_end_id = None
    
    def set_tokenizer(self, tokenizer):
        """tokenizer，token ID"""
        self.tokenizer = tokenizer
        self.encode_token_id = tokenizer.convert_tokens_to_ids(ENCODE_TOKEN)
        self.decode_token_id = tokenizer.convert_tokens_to_ids(DECODE_TOKEN)
        self.planner_token_id = tokenizer.convert_tokens_to_ids(PLANNER_TOKEN)
        self.step_start_id = tokenizer.convert_tokens_to_ids(STEP_START)
        self.step_end_id = tokenizer.convert_tokens_to_ids(STEP_END)
        self.answer_start_id = tokenizer.convert_tokens_to_ids(ANSWER_START)
        self.answer_end_id = tokenizer.convert_tokens_to_ids(ANSWER_END)
        
        # Call subclass hook
        self._on_tokenizer_set()
    
    def _on_tokenizer_set(self):
        """tokenizer"""
        pass
    
    @property
    def device(self):
        """"""
        return next(self.parameters()).device
    
    @abstractmethod
    def forward(self, **kwargs):
        """，"""
        raise NotImplementedError
    
    @abstractmethod
    def inference(self, question: str, **kwargs) -> Tuple[List[str], List[List[str]]]:
        """
        ，
        
        Args:
            question: 
            **kwargs: （max_steps, temperature）
            
        Returns:
            final_answers: 
            reasoning_steps_list: 
        """
        raise NotImplementedError
    
    def _init_backbone(self, model_path: str, vocab_size: int = None, 
                       freeze: bool = False, lora_config: dict = None):
        """
        backbone
        
        Args:
            model_path: 
            vocab_size: （resize）
            freeze: backbone
            lora_config: LoRA
            
        Returns:
            backbone: backbone
        """
        backbone = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Adjust vocab size
        if vocab_size and backbone.config.vocab_size != vocab_size:
            logger.info(f" backbone : {backbone.config.vocab_size} -> {vocab_size}")
            backbone.resize_token_embeddings(vocab_size, mean_resizing=False)
            backbone.config.vocab_size = vocab_size
        
        # Ensure pad_token_id is set
        if backbone.config.pad_token_id is None:
            backbone.config.pad_token_id = backbone.config.eos_token_id
        
        # Apply LoRA
        if lora_config and lora_config.get('enabled', False):
            if not PEFT_AVAILABLE:
                raise ImportError("PEFT library is required for LoRA fine-tuning.")
            
            logger.info("Enabling LoRA for the backbone model...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_config.get('r', 8),
                lora_alpha=lora_config.get('lora_alpha', 16),
                lora_dropout=lora_config.get('lora_dropout', 0.05),
            )
            backbone = get_peft_model(backbone, peft_config)
            backbone.print_trainable_parameters()
        elif freeze:
            logger.info("Freezing LLM Backbone parameters...")
            for param in backbone.parameters():
                param.requires_grad = False
        
        return backbone
    
    def _prepare_question_encoding(self, question: str, max_length: int = 128):
        """
        
        
        Args:
            question: 
            max_length: 
            
        Returns:
            question_ids: (1, seq_len)
            question_mask: (1, seq_len)
        """
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'
        
        question_enc = self.tokenizer(
            question,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        self.tokenizer.padding_side = original_padding_side
        
        return question_enc['input_ids'].to(self.device), question_enc['attention_mask'].to(self.device)
