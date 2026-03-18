# -*- coding: utf-8 -*-
"""
CoTModel - Chain-of-Thought 
CoT， SFT 。
"""

import torch
import torch.nn as nn
import os
from transformers import AutoModelForCausalLM, PreTrainedModel, PretrainedConfig, AutoConfig
from typing import Optional, List, Tuple, Any, Dict
import logging
import re

from plat.config.base_config import (
    CoTSFTConfig,
    STEP_START, STEP_END, ANSWER_START, ANSWER_END
)
from plat.models.base_reasoning_model import BaseReasoningModel, ReasoningOutput
from plat.utils.helpers import extract_answer_from_text

logger = logging.getLogger(__name__)


class CoTModel(BaseReasoningModel):
    """
    Chain-of-Thought  ( SFT)
    """
    
    def __init__(self, config: PretrainedConfig, sft_config: CoTSFTConfig = None):
        super().__init__(config)
        if sft_config is None:
            #  sft_config，（ from_pretrained）
            sft_config = CoTSFTConfig()
            sft_config.vocab_size = config.vocab_size
        self.sft_config = sft_config
        
        # 1. 
        base_name = self.sft_config.sft_model_path or self.sft_config.base_model_name
        logger.info(f"Initializing CoT model from: {base_name}")
        
        # 2.  LoRA 
        lora_cfg = self.sft_config.lora
        lora_dict = {
            'enabled': lora_cfg.enabled,
            'r': lora_cfg.r,
            'lora_alpha': lora_cfg.lora_alpha,
            'lora_dropout': lora_cfg.lora_dropout,
        } if lora_cfg.enabled else None
        
        # 3.  Backbone
        self.backbone = self._init_backbone(
            model_path=base_name,
            vocab_size=self.sft_config.vocab_size,
            freeze=False, # SFT  backbone 
            lora_config=lora_dict
        )
        
        self.hidden_size = self.backbone.config.hidden_size
        logger.info(f"CoT model initialized. Hidden size: {self.hidden_size}")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
         checkpoint 
         custom model （ Hugging Face  auto ）
        """
        import json
        from transformers import AutoModelForCausalLM, PretrainedConfig, GPT2LMHeadModel
        from safetensors import safe_open
        
        # 1.  checkpoint 
        if not os.path.isdir(pretrained_model_name_or_path):
            raise ValueError(f"Checkpoint directory not found: {pretrained_model_name_or_path}")
        
        # 2.  checkpoint  config.json（）
        checkpoint_config_path = os.path.join(pretrained_model_name_or_path, 'config.json')
        if not os.path.exists(checkpoint_config_path):
            raise ValueError(f"config.json not found in {pretrained_model_name_or_path}")
        
        with open(checkpoint_config_path, 'r') as f:
            checkpoint_config = json.load(f)
        
        # 3. （ backbone）
        base_model_type = checkpoint_config.get('model_type', 'gpt2')
        logger.info(f"Checkpoint model_type: {base_model_type}")
        
        # 4. 
        safetensors_path = os.path.join(pretrained_model_name_or_path, 'model.safetensors')
        pytorch_path = os.path.join(pretrained_model_name_or_path, 'pytorch_model.bin')
        
        if not os.path.exists(safetensors_path) and not os.path.exists(pytorch_path):
            raise ValueError(f"No weight file found in {pretrained_model_name_or_path}")
        
        # 5. （ custom ）
        base_config_dict = {k: v for k, v in checkpoint_config.items() 
                           if k not in ['architectures', 'torch_dtype', 'transformers_version']}
        
        #  vocab_size 
        if 'vocab_size' not in base_config_dict:
            base_config_dict['vocab_size'] = 50262
        
        # 6. 
        state_dict = {}
        has_lora = False
        if os.path.exists(safetensors_path):
            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                
                #  LoRA 
                has_lora = any('lora_A' in k or 'lora_B' in k for k in keys)
                logger.info(f"Has LoRA weights: {has_lora}")
                
                # （ backbone. ）
                for key in keys:
                    new_key = key
                    if key.startswith('backbone.'):
                        new_key = key[len('backbone.'):]
                    if 'base_model.model' in new_key:
                        new_key = new_key.replace('base_model.model.', '')
                    state_dict[new_key] = f.get_tensor(key)
        else:
            state_dict = torch.load(pytorch_path, map_location="cpu")
            new_state_dict = {}
            for key in state_dict:
                new_key = key
                if key.startswith('backbone.'):
                    new_key = key[len('backbone.'):]
                if 'base_model.model' in new_key:
                    new_key = new_key.replace('base_model.model.', '')
                new_state_dict[new_key] = state_dict[key]
            state_dict = new_state_dict
        
        # 7.  backbone（ AutoModel，）
        #  checkpoint （AutoConfig  model_type）
        base_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        backbone = AutoModelForCausalLM.from_config(base_config)
        
        logger.info(f"Created backbone: {type(backbone).__name__}")
        
        # 8. 
        missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys when loading backbone: {missing}")
        
        #  tied weights： lm_head.weight ， embed_tokens 
        if 'lm_head.weight' in missing and 'model.embed_tokens.weight' in state_dict:
            backbone.lm_head.weight = backbone.get_input_embeddings().weight
            logger.info(" lm_head.weight  embed_tokens.weight (tied weights)")
        
        logger.info(f"Backbone weights loaded from {pretrained_model_name_or_path}")
        
        # 9.  CoTModel （ backbone）
        logger.info("Creating CoTModel instance...")
        instance = cls.__new__(cls)
        
        #  nn.Module.__init__()（ __init__ ）
        nn.Module.__init__(instance)
        
        #  config  backbone
        instance.config = base_config
        instance.backbone = backbone
        instance.hidden_size = backbone.config.hidden_size
        
        #  sft_config
        instance.sft_config = CoTSFTConfig()
        instance.sft_config.vocab_size = base_config.vocab_size
        
        #  checkpoint  tokenizer
        tokenizer_path = pretrained_model_name_or_path
        if os.path.exists(os.path.join(tokenizer_path, 'tokenizer_config.json')):
            from transformers import AutoTokenizer
            instance.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            logger.info(f"Loaded tokenizer from {tokenizer_path}")
        
        logger.info(f"CoTModel loaded successfully from {pretrained_model_name_or_path}")
        return instance
    
    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: torch.LongTensor,
                labels: torch.LongTensor = None,
                **kwargs) -> ReasoningOutput:
        """SFT """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return ReasoningOutput(
            loss=outputs.loss,
            logits=outputs.logits,
        )
    
    def generate(self, 
                 input_ids: torch.LongTensor = None,
                 attention_mask: torch.LongTensor = None,
                 max_new_tokens: int = 256,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 top_p: float = 1.0,
                 top_k: int = 50,
                 pad_token_id: int = None,
                 eos_token_id: int = None,
                 **kwargs) -> torch.LongTensor:
        """
        ， greedy 
        
        Args:
            input_ids:  token ID
            attention_mask:  mask
            max_new_tokens:  token 
            do_sample: （False=greedy）
            temperature: 
            top_p: top-p 
            top_k: top-k  k 
            pad_token_id: padding token id
            eos_token_id:  token id
        """
        return self.backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
    
    def inference(self, question, **kwargs) -> Tuple[List[str], List[str]]:
        """
        ，
        
        Args:
            question: （str  List[str]）
            **kwargs: （max_new_tokens, do_sample, temperature ）
            
        Returns:
            final_answers: 
            full_generated_texts: 
        """
        self.eval()
        
        # （left padding  batch ）
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'
        
        inputs = self.tokenizer(
            question,
            return_tensors='pt',
            truncation=True,
            max_length=kwargs.get('max_question_length', 256),
            padding=True
        )
        
        self.tokenizer.padding_side = original_padding_side
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        prompt_len = input_ids.shape[1]
        
        # 
        with torch.no_grad():
            outputs = self.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=kwargs.get('max_new_tokens', 256),
                do_sample=kwargs.get('do_sample', False),
                temperature=kwargs.get('temperature', 1.0),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # （，）
        batch_size = outputs.shape[0]
        final_answers = []
        full_texts = []
        
        for b in range(batch_size):
            generated_ids = outputs[b, prompt_len:]
            
            #  pad
            non_pad_mask = (generated_ids != self.tokenizer.pad_token_id)
            if non_pad_mask.any():
                last_non_pad = non_pad_mask.nonzero(as_tuple=True)[0][-1].item()
                generated_ids = generated_ids[:last_non_pad + 1]
            
            full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            final_answer = extract_answer_from_text(full_text)
            
            final_answers.append(final_answer)
            full_texts.append(full_text)
        
        return final_answers, full_texts
    
    @classmethod
    def from_config(cls, sft_config: CoTSFTConfig):
        """ CoTSFTConfig """
        base_name = sft_config.sft_model_path or sft_config.base_model_name
        base_config = PretrainedConfig.from_pretrained(base_name)
        if sft_config.vocab_size:
            base_config.vocab_size = sft_config.vocab_size
        return cls(base_config, sft_config)
