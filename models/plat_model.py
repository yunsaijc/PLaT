# -*- coding: utf-8 -*-

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
from transformers.cache_utils import DynamicCache
from typing import Optional, Tuple, List, Any, Dict
from dataclasses import dataclass
from transformers.utils import ModelOutput
import logging
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, create_causal_mask

from plat.config.base_config import (
    PLaTConfig,
    STEP_START, STEP_END, ANSWER_START, ANSWER_END,
    ENCODE_TOKEN, DECODE_TOKEN, PLANNER_TOKEN
)
from plat.models.base_reasoning_model import BaseReasoningModel, ReasoningOutput
from plat.utils.helpers import (
    truncate_cache, compute_ema_cumulative, extract_answer_from_text
)

logger = logging.getLogger(__name__)


# =========================================================================
#                           Output Data Classes
# =========================================================================

@dataclass
class PLaTOutput(ReasoningOutput):
    """PLaT model output structure"""
    loss_reconstruct: Optional[torch.FloatTensor] = None
    generated_trajectory: Optional[torch.FloatTensor] = None
    generated_ids: Optional[List[torch.LongTensor]] = None

@dataclass
class InferenceOutput(ModelOutput):
    """Inference output structure"""
    latent_trajectory: torch.FloatTensor = None
    decoded_steps: Optional[List[str]] = None
    final_answer: Optional[str] = None
    answer_logits: Optional[torch.FloatTensor] = None


# =========================================================================
#                      BackboneBasedPlanner Definition
# =========================================================================

def _get_backbone_architecture(backbone):
    """Get backbone architecture type from config"""
    config = None
    if hasattr(backbone, 'config'):
        config = backbone.config
    elif hasattr(backbone, 'base_model') and hasattr(backbone.base_model, 'config'):
        config = backbone.base_model.config
    elif hasattr(backbone, 'model') and hasattr(backbone.model, 'config'):
        config = backbone.model.config
    
    if config is not None:
        if hasattr(config, 'model_type'):
            model_type = config.model_type.lower()
            if 'gpt2' in model_type:
                return 'gpt2'
            elif 'qwen' in model_type:
                return 'qwen'
            else: raise ValueError(f"Unknown backbone architecture: {model_type}")
        
        if hasattr(config, 'architectures'):
            arch = config.architectures
            if arch:
                arch_name = arch[0].lower()
                if 'gpt2' in arch_name:
                    return 'gpt2'
    
    raise ValueError(f"Unknown backbone architecture")


def _get_backbone_layers(backbone):
    """Get transformer layers from backbone"""
    if hasattr(backbone, 'transformer') and hasattr(backbone.transformer, 'h'):
        return backbone.transformer.h
    
    if hasattr(backbone, 'model') and hasattr(backbone.model, 'layers'):
        return backbone.model.layers
    
    if hasattr(backbone, 'base_model') and hasattr(backbone.base_model, 'transformer') and hasattr(backbone.base_model.transformer, 'h'):
        return backbone.base_model.transformer.h
    if hasattr(backbone, 'base_model') and hasattr(backbone.base_model, 'model') and hasattr(backbone.base_model.model, 'layers'):
        return backbone.base_model.model.layers
    
    if hasattr(backbone, 'base_model') and hasattr(backbone.base_model, 'model') and hasattr(backbone.base_model.model, 'model'):
        model_inner = backbone.base_model.model.model
        if hasattr(model_inner, 'transformer') and hasattr(model_inner.transformer, 'h'):
            return model_inner.transformer.h
        if hasattr(model_inner, 'layers'):
            return model_inner.layers
    
    raise AttributeError(f"Cannot get backbone layers, type: {type(backbone)}")


def _get_layer_dtype(layer):
    """Get layer dtype"""
    if hasattr(layer, 'ln_1'):
        return layer.ln_1.weight.dtype
    elif hasattr(layer, 'input_layernorm'):
        return layer.input_layernorm.weight.dtype
    else:
        return next(layer.parameters()).dtype


class BackboneBasedPlanner(nn.Module):
    def __init__(self, backbone, hidden_size, latent_dim, use_top_n_layers=-1, 
                 add_extra_layers=0):
        super().__init__()
        
        self.arch = _get_backbone_architecture(backbone)
        logger.info(f"BackboneBasedPlanner: detected architecture '{self.arch}'")
        
        object.__setattr__(self, '_backbone_ref', backbone)
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        
        self.state_to_hidden = nn.Linear(latent_dim, hidden_size)

        core_model = backbone
        if hasattr(core_model, 'base_model'):
            core_model = core_model.base_model
        if hasattr(core_model, 'model') and hasattr(core_model.model, 'layers'):
            core_model = core_model.model

        all_layers = _get_backbone_layers(backbone)
        if use_top_n_layers == -1:
            object.__setattr__(self, 'backbone_layers', all_layers)
        else:
            object.__setattr__(self, 'backbone_layers', all_layers[-use_top_n_layers:])

        self.extra_layers = None
        self.rotary_emb = None
        
        if add_extra_layers > 0:
            if self.arch == 'gpt2':
                self.extra_layers = nn.ModuleList([
                    GPT2Block(backbone.config, layer_idx=len(all_layers) + i)
                    for i in range(add_extra_layers)
                ])
        
        layer_norm_eps = getattr(backbone.config, 'layer_norm_epsilon', 
                                  getattr(backbone.config, 'rms_norm_eps', 1e-5))
        self.ln_f = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        self.hidden_to_state = nn.Sequential(
            nn.Linear(hidden_size, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        self.planner_token_id = None
    
    def _get_dtype(self):
        """Get layer dtype"""
        return _get_layer_dtype(self.backbone_layers[0])

    def forward(self, question_embeds, question_state, state_history, past_key_values=None):
        batch_size = question_state.size(0)
        device = question_state.device
        backbone_dtype = self._get_dtype()
        
        if past_key_values is None:
            q_state_proj = self.state_to_hidden(question_state.unsqueeze(1).to(backbone_dtype))
            history_proj = self.state_to_hidden(state_history.to(backbone_dtype))
            
            planner_token_ids = torch.full((batch_size, 1), self.planner_token_id, dtype=torch.long, device=device)
            mode_token = self._backbone_ref.get_input_embeddings()(planner_token_ids).to(backbone_dtype)
            hidden_states = torch.cat([mode_token, q_state_proj, history_proj], dim=1)
                
            if question_embeds is not None:
                hidden_states = torch.cat([question_embeds.to(backbone_dtype), hidden_states], dim=1)
            
            past_key_values = DynamicCache()
        else:
            new_state = state_history[:, -1:, :]
            hidden_states = self.state_to_hidden(new_state.to(backbone_dtype))

        seq_len = hidden_states.size(1)
        past_seen_tokens = past_key_values.get_seq_length()
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + seq_len, device=device)
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
        
        position_embeddings = None
        
        hidden_states, causal_mask = self._forward_gpt2_style(
            hidden_states, position_ids, cache_position, past_key_values
        )
        
        for layer in self.backbone_layers:
            outputs = layer(
                hidden_states,
                past_key_values=past_key_values,
                cache_position=cache_position,
                attention_mask=causal_mask,
                use_cache=True,
            )
            hidden_states = outputs[0]
        
        if self.extra_layers:
            for layer in self.extra_layers:
                outputs = layer(
                    hidden_states,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    attention_mask=causal_mask,
                    use_cache=True,
                )
                hidden_states = outputs[0]
                
        hidden_states = self.ln_f(hidden_states)
        last_hidden = hidden_states[:, -1, :]
        next_state = self.hidden_to_state(last_hidden)
        
        return next_state, past_key_values

    def _forward_gpt2_style(self, hidden_states, position_ids, cache_position, past_key_values):
        """GPT-2 style forward processing"""
        base = self._backbone_ref
        
        if hasattr(base, 'base_model'):
            base = base.base_model
        if hasattr(base, 'model') and hasattr(base.model, 'model'):
            base = base.model
        if hasattr(base, 'model') and not hasattr(base, 'transformer'):
            base = base.model
        
        if hasattr(base, 'wpe') and hasattr(base, 'drop'):
            position_embeds = base.wpe(position_ids)
            hidden_states = hidden_states + position_embeds
            hidden_states = base.drop(hidden_states)
        else:
            position_embeds = base.transformer.wpe(position_ids)
            hidden_states = hidden_states + position_embeds
            hidden_states = base.transformer.drop(hidden_states)
        
        causal_mask = create_causal_mask(
            config=self._backbone_ref.config,
            input_embeds=hidden_states,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        
        return hidden_states, causal_mask


class PLaTModel(BaseReasoningModel):
    config_class = PretrainedConfig
    
    def __init__(self, 
                 config: PretrainedConfig, 
                 plat_config: PLaTConfig):
        # Call BaseReasoningModel __init__
        super().__init__(config)
        self.plat_config = plat_config
        model_cfg = plat_config.model
        train_cfg = plat_config.train
        lora_cfg = plat_config.lora

        
        # Load LLM Backbone
        base_name = model_cfg.sft_model_path or model_cfg.base_model_name
        logger.info(f"Loading LLM Backbone from: {base_name}")
        self.backbone = AutoModelForCausalLM.from_pretrained(base_name)
        self.arch = _get_backbone_architecture(self.backbone)
        
        # Adjust vocab size if needed
        if model_cfg.vocab_size and self.backbone.config.vocab_size != model_cfg.vocab_size:
            logger.info(f"Adjust backbone vocab size: {self.backbone.config.vocab_size} -> {model_cfg.vocab_size}")
            self.backbone.resize_token_embeddings(model_cfg.vocab_size, mean_resizing=False)
            self.backbone.config.vocab_size = model_cfg.vocab_size
        
        # Ensure pad_token_id is set
        if self.backbone.config.pad_token_id is None:
            self.backbone.config.pad_token_id = self.backbone.config.eos_token_id
        
        # Get hidden size
        hidden_size = self.backbone.config.hidden_size
        latent_dim = model_cfg.latent_dim
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        
        # 2. LoRA config
        self.freeze_backbone = train_cfg.freeze_backbone
        
        if lora_cfg.enabled:
            
            logger.info("Enabling LoRA for the backbone model...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_cfg.r,
                lora_alpha=lora_cfg.lora_alpha,
                lora_dropout=lora_cfg.lora_dropout,
            )
            self.backbone = get_peft_model(self.backbone, peft_config)
            self.backbone.print_trainable_parameters()
        elif train_cfg.freeze_backbone:
            logger.info("Freezing LLM Backbone parameters...")
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            logger.info("LLM Backbone will be fine-tuned...")
        
        # 3. Independent decoder (optional)
        self.use_independent_decoder = model_cfg.use_independent_decoder
        if self.use_independent_decoder:
            self.decoder_lm = AutoModelForCausalLM.from_pretrained(base_name)
            if model_cfg.vocab_size and self.decoder_lm.config.vocab_size != model_cfg.vocab_size:
                self.decoder_lm.resize_token_embeddings(model_cfg.vocab_size, mean_resizing=False)
            if self.decoder_lm.config.pad_token_id is None:
                self.decoder_lm.config.pad_token_id = self.decoder_lm.config.eos_token_id
            
            if lora_cfg.enabled:
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_cfg.r,
                    lora_alpha=lora_cfg.lora_alpha,
                )
                self.decoder_lm = get_peft_model(self.decoder_lm, peft_config)
        
        # 4. State Encoder
        encoder_cfg = model_cfg.encoder
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # 5. Latent Thought Model
        planner_cfg = model_cfg.planner
        planner_type = planner_cfg.type
        
        if planner_type == 'head':
            # Planner Head mode
            logger.info("Initialize Planner Head")
            self.planner_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, latent_dim)
            )
            self.planner = None
            self.planner_type = 'head'
        elif planner_type == 'backbone_based':
            logger.info("Initialize Backbone-based Planner")
            self.planner = BackboneBasedPlanner(
                backbone=self.backbone,
                hidden_size=hidden_size,
                latent_dim=latent_dim,
                use_top_n_layers=planner_cfg.use_top_n_layers,
                add_extra_layers=planner_cfg.add_extra_layers,
            )
            self.planner_type = 'backbone_based'
        else:
            raise ValueError(f"Unknown planner type: {planner_type}")
        
        # 6. Number of latents per step
        self.num_latent_for_a_step = max(1, model_cfg.num_latent_for_a_step)
        
        # 7. Decoder projection
        self.decode_proj = nn.Linear(latent_dim, hidden_size)
        
        # Special token IDs
        self.encode_token_id = None
        self.decode_token_id = None
        self.planner_token_id = None
        self.step_start_id = None
        self.step_end_id = None
        self.answer_start_id = None
        self.answer_end_id = None
        self.tokenizer = None
        
        # Note: Do not create Accelerator inside model
        # This causes significant memory overhead
        self._accelerator = None
        
        # Dual LoRA state
        self.dual_lora_enabled = False
        
        logger.info(f"PLaT model initialized: latent_dim={latent_dim}, num_latent_per_step={self.num_latent_for_a_step}")

    def _on_tokenizer_set(self):
        """
        Override parent method: additional logic after tokenizer set
        Set planner token id
        """
        if hasattr(self, 'planner') and self.planner is not None:
            self.planner.planner_token_id = self.planner_token_id

    # =========================================================================
    #                         Model Architecture Compatibility Layer
    # =========================================================================
    
    def _get_backbone_transformer_layers(self):
        """Get transformer layers from backbone"""
        backbone = self.backbone
        if hasattr(backbone, 'transformer') and hasattr(backbone.transformer, 'h'):
            return backbone.transformer.h
        if hasattr(backbone, 'model') and hasattr(backbone.model, 'layers'):
            return backbone.model.layers
        if hasattr(backbone, 'layers'):
            return backbone.layers
        
        if hasattr(backbone, 'base_model'):
            base = backbone.base_model
            if hasattr(base, 'transformer') and hasattr(base.transformer, 'h'):
                return base.transformer.h
            if hasattr(base, 'model') and hasattr(base.model, 'layers'):
                return base.model.layers
            if hasattr(base, 'model') and hasattr(base.model, 'model'):
                inner = base.model.model
                if hasattr(inner, 'transformer') and hasattr(inner.transformer, 'h'):
                    return inner.transformer.h
                if hasattr(inner, 'layers'):
                    return inner.layers
        
        raise AttributeError("Unrecognized backbone architecture")
    
    def _get_backbone_layer_norm(self, layer_idx: int = 0):
        """Get layer norm for dtype"""
        layers = self._get_backbone_transformer_layers()
        layer = layers[layer_idx]
        
        if hasattr(layer, 'ln_1'):
            return layer.ln_1
        else:
            return None
    
    def _get_backbone_dtype(self):
        """Get backbone dtype with caching"""
        if hasattr(self, '_cached_backbone_dtype') and self._cached_backbone_dtype is not None:
            return self._cached_backbone_dtype
        ln = self._get_backbone_layer_norm(0)
        if ln is not None and hasattr(ln, 'weight'):
            self._cached_backbone_dtype = ln.weight.dtype
            return self._cached_backbone_dtype
        # Fallback: use first param dtype
        self._cached_backbone_dtype = next(self.backbone.parameters()).dtype
        return self._cached_backbone_dtype
    
    def _get_backbone_final_norm(self):
        """
        Get backbone final LayerNorm
        
        Returns:
            nn.LayerNorm or similar
        """
        # Handle PEFT wrapping
        base = self.backbone
        if hasattr(base, 'base_model'):
            base = base.base_model
        if hasattr(base, 'model') and hasattr(base.model, 'model'):
            base = base.model
        
        # GPT-2: transformer.ln_f
        if hasattr(base, 'transformer') and hasattr(base.transformer, 'ln_f'):
            return base.transformer.ln_f
        # LLaMA/Qwen: model.norm
        elif hasattr(base, 'model') and hasattr(base.model, 'norm'):
            return base.model.norm
        # Has norm directly
        elif hasattr(base, 'norm'):
            return base.norm
        else:
            return None

    # =========================================================================
    #                             Dual LoRA Support
    # =========================================================================
    
    def prepare_for_rl_with_dual_lora(self, planner_adapter_name='planner', 
                                       decoder_adapter_name='decoder', 
                                       freeze_planner_lora=True):
        """
        Prepare Dual LoRA mode for RL training
        
        1. Copy LoRA adapter to planner adapter
        2. Copy LoRA adapter to decoder adapter
        3. Freeze planner head / extra layers
        
        Args:
            planner_adapter_name: name of planner adapter
            decoder_adapter_name: name of decoder adapter
            freeze_planner_lora: whether to freeze planner LoRA
        """

        planner_status = "Frozen" if freeze_planner_lora else "Trainable"
        logger.info(f"Preparing Dual LoRA for RL: Planner='{planner_adapter_name}' ({planner_status}), Decoder='{decoder_adapter_name}' (Trainable)")
        
        lora_cfg = self.plat_config.lora
        
        # Check if backbone is already wrapped by PEFT
        if not hasattr(self.backbone, 'peft_config'):
            logger.info("Backbone not wrapped by PEFT, initializing...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_cfg.r,
                lora_alpha=lora_cfg.lora_alpha,
                lora_dropout=lora_cfg.lora_dropout,
            )
            self.backbone = get_peft_model(self.backbone, peft_config)
            logger.info("Initialized PEFT adapter for backbone")
        
        # Get base adapter
        base_adapter = 'default'
        if base_adapter not in self.backbone.peft_config:
            if len(self.backbone.peft_config) > 0:
                base_adapter = next(iter(self.backbone.peft_config.keys()))
        
        logger.info(f"Base adapter: {base_adapter}")
        
        # 1. Create Planner Adapter
        if planner_adapter_name not in self.backbone.peft_config:
            self.backbone.add_adapter(planner_adapter_name, self.backbone.peft_config[base_adapter])
            self._copy_adapter_weights(base_adapter, planner_adapter_name)
        
        # 2. Create Decoder Adapter
        if decoder_adapter_name not in self.backbone.peft_config:
            self.backbone.add_adapter(decoder_adapter_name, self.backbone.peft_config[base_adapter])
            self._copy_adapter_weights(base_adapter, decoder_adapter_name)

        # Set active adapter to decoder
        self.backbone.set_adapter(decoder_adapter_name)

        # 3. Set LoRA gradients
        for n, p in self.backbone.named_parameters():
            if planner_adapter_name in n:
                p.requires_grad = not freeze_planner_lora
            elif decoder_adapter_name in n:
                p.requires_grad = True
            elif "lora_" in n:
                p.requires_grad = False
                
        # 4. Freeze Planner parts
        for p in self.encoder.parameters():
            p.requires_grad = False
            
        if self.planner_type == 'backbone_based' and self.planner is not None:
            if hasattr(self.planner, 'extra_layers') and self.planner.extra_layers:
                for p in self.planner.extra_layers.parameters():
                    p.requires_grad = False
            if hasattr(self.planner, 'state_to_hidden'):
                for p in self.planner.state_to_hidden.parameters():
                    p.requires_grad = False
            if hasattr(self.planner, 'hidden_to_state'):
                for p in self.planner.hidden_to_state.parameters():
                    p.requires_grad = False
            if hasattr(self.planner, 'ln_f'):
                for p in self.planner.ln_f.parameters():
                    p.requires_grad = False
            if hasattr(self.planner, 'output_norm'):
                for p in self.planner.output_norm.parameters():
                    p.requires_grad = False
            if hasattr(self.planner, 'mode_token_embed'):
                self.planner.mode_token_embed.requires_grad = False
        elif self.planner_type == 'head':
            for p in self.planner_head.parameters():
                p.requires_grad = False
                
        # 5. Record state
        self.dual_lora_enabled = True
        self.planner_adapter_name = planner_adapter_name
        self.decoder_adapter_name = decoder_adapter_name
        
        logger.info("Dual LoRA setup complete.")
    
    def _copy_adapter_weights(self, source_adapter_name, target_adapter_name):
        """
        Copy adapter weights from source to target
        
        Args:
            source_adapter_name: source adapter name
            target_adapter_name: target adapter name
        """
        
        source_state_dict = get_peft_model_state_dict(self.backbone, adapter_name=source_adapter_name)
        
        target_state_dict = {}
        for key, value in source_state_dict.items():
            target_key = key.replace(f".{source_adapter_name}.", f".{target_adapter_name}.")
            target_state_dict[target_key] = value
        
        set_peft_model_state_dict(self.backbone, target_state_dict, adapter_name=target_adapter_name)
        
        logger.info(f"Copied adapter weights: {source_adapter_name} -> {target_adapter_name}")
        
    def set_adapter_mode(self, mode):
        """
        Switch adapter mode
        
        Args:
            mode: 'planner' or 'decoder'
        """
        if not getattr(self, 'dual_lora_enabled', False):
            return
            
        if mode == 'planner':
            self.backbone.set_adapter(self.planner_adapter_name)
        elif mode == 'decoder':
            self.backbone.set_adapter(self.decoder_adapter_name)
        else:
            logger.warning(f"Unknown adapter mode: {mode}, available modes: 'planner', 'decoder'")

    # =========================================================================
    #                             Encoder: Context → State
    # =========================================================================
    
    def encode_state(self, context_ids, context_mask):
        """
        Encode context to latent state
        
        Args:
            context_ids: (batch_size, seq_len)
            context_mask: (batch_size, seq_len)
            
        Returns:
            state: (batch_size, latent_dim)
        """
        with torch.set_grad_enabled(not self.freeze_backbone):
            last_token_hidden = self.backbone(
                input_ids=context_ids,
                attention_mask=context_mask,
                return_dict=True,
                output_hidden_states=True
            ).hidden_states[-1][:, -1, :]  # (batch_size, hidden_size)
        state = self.encoder(last_token_hidden)  # (batch_size, latent_dim)
        return state

    # =========================================================================
    #                             Planner: State → Next State
    # =========================================================================
    
    def planner_forward_with_head(self, question_embeds, question_state, 
                                   state_history, past_key_values=None):
        """
        Predict next state using backbone + planner_head
        """
        batch_size = question_state.size(0)
        device = question_state.device
        weight_dtype = self.decode_proj.weight.dtype
        backbone_dtype = self._get_backbone_dtype()
        
        if past_key_values is None:
            # First call: [<PLANNER>, s_0]
            planner_token_ids = torch.full((batch_size, 1), self.planner_token_id, 
                                       dtype=torch.long, device=device)
            planner_token_embed = self.backbone.get_input_embeddings()(planner_token_ids).to(backbone_dtype)
            
            question_state_proj = self.decode_proj(question_state.to(weight_dtype).unsqueeze(1)).to(backbone_dtype)
            
            if question_embeds is not None:
                inputs_embeds = torch.cat([question_embeds, planner_token_embed, question_state_proj], dim=1)
            else:
                inputs_embeds = torch.cat([planner_token_embed, question_state_proj], dim=1)
            seq_len = inputs_embeds.size(1)
            
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            position_embeds = self.backbone.transformer.wpe(position_ids).to(backbone_dtype)
            inputs_embeds = inputs_embeds + position_embeds
        else:
            # Subsequent calls: only input new state
            new_state = state_history[:, -1:, :]
            new_state_proj = self.decode_proj(new_state.to(weight_dtype)).to(backbone_dtype)
            
            inputs_embeds = new_state_proj
            
            past_length = past_key_values[0][0].size(2) if past_key_values[0][0].dim() == 4 else past_key_values[0][0].size(-2)
            position_ids = torch.arange(past_length, past_length + 1, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            position_embeds = self.backbone.transformer.wpe(position_ids).to(backbone_dtype)
            inputs_embeds = inputs_embeds + position_embeds
        
        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = outputs.hidden_states[-1]
        pred_hidden = hidden_states[:, -1, :]
        next_state = self.planner_head(pred_hidden)
        
        return next_state, outputs.past_key_values

    # =========================================================================
    #                             Decoder: State → Text
    # =========================================================================
    
    def decode_state(self, 
                     state_chunks=None, 
                     prefix_past_kv=None, 
                     target_ids=None, 
                     target_mask=None, 
                     labels=None, 
                     decode_mode='train', 
                     return_logits=False, 
                     keep_logits_grad=False,
                     stop_token_ids=None, 
                     max_length=64, 
                     temperature=1.0,
                     do_sample=False,
                     return_cache=False):
        """
        Decode from latent state to text
        
        Args:
            state_chunks: (batch_size, num_latent_per_step, latent_dim)
            prefix_past_kv: prefix KV cache
            target_ids: target sequence token IDs
            target_mask: (batch_size, seq_len) attention mask
            decode_mode: 'train' or 'generate'
            
        Returns:
            train mode: (logits, loss)
            generate mode: (generated_ids, all_logits)
        """
        batch_size = state_chunks.size(0)
        device = state_chunks.device
        num_latent_tokens = state_chunks.size(1)
        
        backbone_dtype = self._get_backbone_dtype()
        embedding_layer = self.backbone.get_input_embeddings()
        
        # Construct prefix sequence
        prefix_components = []
        prefix_mask_components = []
        
        
        # Add <DECODE> token
        decode_token_ids = torch.full((batch_size, 1), self.decode_token_id, dtype=torch.long, device=device)
        decode_embed = embedding_layer(decode_token_ids).to(backbone_dtype)
        prefix_components.append(decode_embed)
        prefix_mask_components.append(torch.ones((batch_size, 1), dtype=torch.long, device=device))
        
        # Concatenate prefix
        prefix_embeds = torch.cat(prefix_components, dim=1)
        prefix_mask = torch.cat(prefix_mask_components, dim=1)
        prefix_len = prefix_embeds.size(1)
        
        # Add position encoding
        cache_length = prefix_past_kv[0][0].size(-2) if prefix_past_kv is not None else 0
        position_ids = torch.arange(cache_length, cache_length + prefix_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        position_embeds = self.backbone.transformer.wpe(position_ids).to(backbone_dtype)
        prefix_embeds = prefix_embeds + position_embeds
        
        # Forward prefix, build KV cache
        decoder_model = self.decoder_lm if self.use_independent_decoder else self.backbone
        outputs = decoder_model(
            inputs_embeds=prefix_embeds,
            attention_mask=prefix_mask,
            past_key_values=prefix_past_kv,
            use_cache=True,
            return_dict=True
        )
        current_past_kv = outputs.past_key_values
        
        # Project state to LLM space
        current_state_embed = self.decode_proj(state_chunks)
        if decode_mode == 'train':
            return self._decode_train(
                current_state_embed, current_past_kv, target_ids, target_mask,
                batch_size, device, backbone_dtype, embedding_layer,
                return_logits, keep_logits_grad, decoder_model
            )
        elif decode_mode == 'generate':
            return self._decode_generate(
                current_state_embed, current_past_kv, stop_token_ids,
                batch_size, device, backbone_dtype, embedding_layer,
                max_length, temperature, do_sample, decoder_model,
                prefix_mask, return_cache
            )
        else:
            raise ValueError(f"Unknown decode_mode: {decode_mode}")

    def _decode_train(self, current_state_embed, current_past_kv, target_ids, target_mask,
                      batch_size, device, backbone_dtype, embedding_layer,
                      return_logits, keep_logits_grad, decoder_model):
        """Decode in training mode"""
        if target_ids is None:
            raise ValueError("target_ids must be provided in 'train' mode.")
        
        seq_len = target_ids.size(1)
        num_latent_tokens = current_state_embed.size(1)
        
        # BOS token
        bos_token_id = self.backbone.config.pad_token_id
        bos_embed = embedding_layer(
            torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        ).to(backbone_dtype)
        
        # Target token embedding
        target_ids_for_embed = target_ids.clone()
        target_ids_for_embed[target_ids_for_embed == -100] = bos_token_id
        decoder_embeds = embedding_layer(target_ids_for_embed).to(backbone_dtype)
        
        # Concatenate states + BOS + target
        decoder_inputs_embeds = torch.cat([current_state_embed, bos_embed, decoder_embeds], dim=1)
        
        # Add position encoding
        current_past_length = current_past_kv[0][0].size(-2)
        decoder_position_ids = torch.arange(
            current_past_length, 
            current_past_length + num_latent_tokens + seq_len + 1, 
            dtype=torch.long, device=device
        )
        decoder_position_ids = decoder_position_ids.unsqueeze(0).expand(batch_size, -1)
        
        decoder_position_embeds = self.backbone.transformer.wpe(decoder_position_ids).to(backbone_dtype)
        decoder_inputs_embeds = decoder_inputs_embeds + decoder_position_embeds
        
        # Construct labels
        labels_full = torch.full(
            (batch_size, num_latent_tokens + seq_len + 1), 
            -100, 
            dtype=target_ids.dtype, 
            device=device
        )
        labels_full[:, num_latent_tokens + 1:] = target_ids
        
        # Construct attention mask
        if target_mask is not None:
            decoder_mask = torch.cat([
                torch.ones((batch_size, num_latent_tokens + 1), dtype=target_mask.dtype, device=device),
                target_mask
            ], dim=1)
            cache_mask = torch.ones((batch_size, current_past_length), dtype=target_mask.dtype, device=device)
            attn_mask_full = torch.cat([cache_mask, decoder_mask], dim=1)
        else:
            attn_mask_full = None
        
        # Forward
        outputs = decoder_model(
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=attn_mask_full,
            past_key_values=current_past_kv,
            labels=labels_full,
            use_cache=False,
            return_dict=True
        )
        
        logits = outputs.logits[:, num_latent_tokens:, :]
        loss = outputs.loss
        
        if return_logits:
            if not keep_logits_grad:
                logits = logits.detach().cpu()
            return logits, loss
        else:
            return None, loss

    def _decode_generate(self, current_state_embed, current_past_kv, stop_token_ids,
                         batch_size, device, backbone_dtype, embedding_layer,
                         max_length, temperature, do_sample, decoder_model,
                         prefix_mask, return_cache):
        """Decode in generation mode"""
        if stop_token_ids is None:
            raise ValueError("stop_token_ids must be provided in 'generate' mode.")
        
        if isinstance(stop_token_ids, list):
            stop_token_ids = torch.tensor(stop_token_ids, device=device, dtype=torch.long)
        else:
            stop_token_ids = stop_token_ids.to(device)
        
        num_latent_tokens = current_state_embed.size(1)
        
        # Generate start token must match training
        start_token_id = self.backbone.config.pad_token_id
        if start_token_id is None and self.tokenizer is not None:
            start_token_id = self.tokenizer.pad_token_id
        if start_token_id is None:
            start_token_id = self.backbone.config.eos_token_id
        
        generated_ids = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
        all_logits = []
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Initialize attention mask
        cache_len = current_past_kv[0][0].size(-2)
        current_attention_mask = torch.ones((batch_size, cache_len), dtype=torch.long, device=device)
        
        for step in range(max_length):
            if step == 0:
                # First step: forward states + BOS
                current_token_ids = generated_ids
                bos_embeds = embedding_layer(current_token_ids).to(backbone_dtype)
                inputs_embeds = torch.cat([current_state_embed, bos_embeds], dim=1)
                
                current_past_length = current_past_kv[0][0].size(-2)
                position_ids = torch.arange(
                    current_past_length, 
                    current_past_length + num_latent_tokens + 1,
                    dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

                if self.arch == 'gpt2':
                    position_embeds = self.backbone.transformer.wpe(position_ids).to(backbone_dtype)
                    inputs_embeds = inputs_embeds + position_embeds
                
                step_mask = torch.ones((batch_size, num_latent_tokens + 1), dtype=torch.long, device=device)
                current_attention_mask = torch.cat([current_attention_mask, step_mask], dim=1)
            else:
                # Subsequent steps: only forward current token
                current_token_ids = generated_ids[:, -1:]
                inputs_embeds = embedding_layer(current_token_ids).to(backbone_dtype)
                
                current_past_length = current_past_kv[0][0].size(-2)
                position_ids = torch.full((batch_size, 1), current_past_length, dtype=torch.long, device=device)

                if self.arch == 'gpt2':
                    position_embeds = self.backbone.transformer.wpe(position_ids).to(backbone_dtype)
                    inputs_embeds = inputs_embeds + position_embeds
                
                current_mask_step = torch.ones((batch_size, 1), dtype=torch.long, device=device)
                current_attention_mask = torch.cat([current_attention_mask, current_mask_step], dim=1)
            
            outputs = decoder_model(
                inputs_embeds=inputs_embeds,
                attention_mask=current_attention_mask,
                past_key_values=current_past_kv,
                use_cache=True,
                return_dict=True
            )
            
            logits = outputs.logits[:, -1, :]
            current_past_kv = outputs.past_key_values
            all_logits.append(logits.unsqueeze(1))
            
            # Generate next token
            if do_sample:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            # Fill finished sequences with pad_token
            next_token = torch.where(
                is_finished.unsqueeze(-1), 
                torch.full_like(next_token, self.tokenizer.pad_token_id), 
                next_token
            )
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Check stop condition
            next_ids = next_token.squeeze(-1)
            for stop_id in stop_token_ids:
                is_finished |= (next_ids == stop_id)
            
            if is_finished.all():
                break
        
        all_logits = torch.cat(all_logits, dim=1)
        generated_ids = generated_ids[:, 1:]  # remove BOS
        
        if return_cache:
            return generated_ids, all_logits, current_past_kv
        else:
            return generated_ids, all_logits


    # =========================================================================
    #                             Forward Pass
    # =========================================================================

    def forward(self, 
                question_input_ids: torch.LongTensor,
                question_attention_mask: torch.LongTensor,
                targets_input_ids: List[torch.LongTensor],
                targets_attention_mask: List[torch.LongTensor] = None,
                training_state: Any = None,
                **kwargs) -> PLaTOutput:

        # Get training info from training_state
        current_epoch = getattr(training_state, 'current_epoch', 0)
        global_step = getattr(training_state, 'global_step', None)
        eval_mode = getattr(training_state, 'is_eval', False)
        
        # Log output control
        # Control log frequency
        log_interval = 20 if eval_mode else 50
        should_log_debug = (global_step is not None and global_step % log_interval == 0) if global_step is not None else True
        # Only log on main process
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        should_log_debug = should_log_debug and is_main_process
        
        # Mode prefix for log
        mode_prefix = "[EVAL]" if eval_mode else "[TRAIN]"
        
        train_cfg = self.plat_config.train
        model_cfg = self.plat_config.model
        
        num_steps = len(targets_input_ids)
        batch_size = question_input_ids.size(0)
        device = question_input_ids.device
        
        if should_log_debug:
            logger.debug(f"\n\n\n{'='*20} {mode_prefix} PLaT FORWARD PASS {'='*20}")
            logger.debug(f"{mode_prefix} Batch size: {batch_size}, Num target steps: {num_steps}, Current epoch: {current_epoch}")
        
        # Step 1: Encode question to get initial state s_0
        encode_token_tensor = torch.full((batch_size, 1), self.encode_token_id, 
                                         dtype=question_input_ids.dtype, device=device)
        encode_attention_tensor = torch.ones((batch_size, 1), 
                                             dtype=question_attention_mask.dtype, device=device)
        question_input_ids_enc = torch.cat([question_input_ids, encode_token_tensor], dim=1)
        question_attention_mask_enc = torch.cat([question_attention_mask, encode_attention_tensor], dim=1)
        s_0 = self.encode_state(question_input_ids_enc, question_attention_mask_enc)
        
        if should_log_debug:
            logger.debug(f"{mode_prefix} Encoded question to initial state s_0")
            question_ids_for_print = question_input_ids_enc[0]
            logger.debug(f"{mode_prefix} [Question]: {self.tokenizer.decode(question_ids_for_print, skip_special_tokens=False).strip('<|endoftext|>')}")
        
        # Step 2: Prepare question embeddings
        question_embeds = self.backbone.get_input_embeddings()(question_input_ids)
        backbone_dtype = self._get_backbone_dtype()
        question_embeds = question_embeds.to(backbone_dtype)
        
        # Step 3: Dynamic generation + instant decoding
        states_list = [s_0]
        past_key_values = None
        num_latent_per_step = self.num_latent_for_a_step
        
        ema_alpha = train_cfg.ema_alpha
        state_transition_mode = train_cfg.state_transition_mode
        add_dyn_ques_context = train_cfg.add_dyn_ques_context
        
        # Latent Denoising
        denoising_cfg = train_cfg.latent_denoising
        use_denoising = denoising_cfg.enabled and not eval_mode
        current_noise_std = 0.0
        if use_denoising:
            if current_epoch >= denoising_cfg.start_epoch:
                if denoising_cfg.noise_schedule == 'constant':
                    current_noise_std = denoising_cfg.noise_std
                elif current_epoch <= denoising_cfg.end_epoch:
                    decay_ratio = (current_epoch - denoising_cfg.start_epoch) / max(1, denoising_cfg.end_epoch - denoising_cfg.start_epoch)
                    if denoising_cfg.noise_schedule == 'linear_decay':
                        current_noise_std = denoising_cfg.noise_std - decay_ratio * (denoising_cfg.noise_std - denoising_cfg.min_noise_std)
                    elif denoising_cfg.noise_schedule == 'exponential_decay':
                        current_noise_std = denoising_cfg.noise_std * (denoising_cfg.min_noise_std / denoising_cfg.noise_std) ** decay_ratio
                else:
                    current_noise_std = denoising_cfg.min_noise_std
        
        if should_log_debug:
            if use_denoising:
                logger.debug(f"{mode_prefix} [Denoising] Enabled with noise_std={current_noise_std:.4f}")
        
        def _generate_next_state():
            nonlocal past_key_values
            if len(states_list) > 1:
                history = torch.stack(states_list[1:], dim=1)
                
                # Denoising
                if use_denoising and current_noise_std > 0:
                    noise = torch.randn_like(history) * current_noise_std
                    history = history + noise
            else:
                history = torch.empty(batch_size, 0, self.latent_dim, device=device)
            
            if self.planner_type == 'head':
                s_t, past_key_values_inner = self.planner_forward_with_head(
                    question_embeds if add_dyn_ques_context else None, 
                    s_0, history, past_key_values
                )
            else:
                s_t, past_key_values_inner = self.planner(
                    question_embeds if add_dyn_ques_context else None,
                    s_0, history, past_key_values=past_key_values
                )
            
            # Truncate gradient graph
            if past_key_values_inner is not None:
                if isinstance(past_key_values_inner, DynamicCache):
                    for _layer in past_key_values_inner.layers:
                        _layer.keys = _layer.keys.detach()
                        _layer.values = _layer.values.detach()
                elif isinstance(past_key_values_inner, tuple):
                    past_key_values_inner = tuple(
                        tuple(kv.detach() if kv is not None else None for kv in layer_past)
                        for layer_past in past_key_values_inner
                    )
            past_key_values = past_key_values_inner
            
            # Add noise to new state
            if use_denoising and current_noise_std > 0 and not denoising_cfg.apply_to_history_only:
                noise_new = torch.randn_like(s_t) * current_noise_std
                s_t = s_t + noise_new
            
            states_list.append(s_t)
            return s_t
        
        # Initialize slot accumulators
        slot_accumulators = [None] * num_latent_per_step
        slot_accumulators[0] = s_0
        for slot_idx in range(1, num_latent_per_step):
            s_extra = _generate_next_state()
            slot_accumulators[slot_idx] = s_extra
        
        total_loss_reconstruct = None
        total_weight_reconstruct = 0.0
        generated_logits = None
        
        for step_idx in range(num_steps):
            state_chunks_for_decode = None
            should_skip_generation = (step_idx == 0)
            
            # Generate and accumulate slot states
            if not should_skip_generation:
                for slot_idx in range(num_latent_per_step):
                    s_new = _generate_next_state()
                    if slot_accumulators[slot_idx] is None:
                        slot_accumulators[slot_idx] = s_new
                    else:
                        if state_transition_mode == 'ema':
                            slot_accumulators[slot_idx] = ema_alpha * s_new + (1 - ema_alpha) * slot_accumulators[slot_idx]
                        elif state_transition_mode == 'residual':
                            slot_accumulators[slot_idx] = slot_accumulators[slot_idx] + s_new
                        elif state_transition_mode == 'independent':
                            slot_accumulators[slot_idx] = s_new
            
            state_chunks_for_decode = torch.stack(slot_accumulators, dim=1)            
            
            if eval_mode:
                # Eval mode: autoregressive decode
                stop_token_ids = [self.step_end_id, self.answer_end_id, self.tokenizer.eos_token_id]
                stop_token_ids = [tid for tid in stop_token_ids if tid is not None]
                
                generated_ids, logits_t = self.decode_state(
                    state_chunks=state_chunks_for_decode,
                    decode_mode='generate',
                    stop_token_ids=stop_token_ids,
                    max_length=targets_input_ids[step_idx].size(1) + 10,
                    do_sample=False
                )
                
                # Keep last step raw logits
                # Return raw logits
                if step_idx == num_steps - 1:
                    generated_logits = logits_t
                
                # Align length for loss
                target_len = targets_input_ids[step_idx].size(1)
                gen_len = logits_t.size(1)
                
                if gen_len < target_len:
                    logits_t = F.pad(logits_t, (0, 0, 0, target_len - gen_len), value=0.0)
                else:
                    logits_t = logits_t[:, :target_len, :]
                
                labels_aligned = targets_input_ids[step_idx].clone()
                loss_decode_t = F.cross_entropy(
                    logits_t.contiguous().view(-1, logits_t.size(-1)),
                    labels_aligned.contiguous().view(-1),
                    ignore_index=-100
                )
            else:
                logits_t, loss_decode_t = self.decode_state(
                    state_chunks=state_chunks_for_decode,
                    target_ids=targets_input_ids[step_idx],
                    target_mask=targets_attention_mask[step_idx] if targets_attention_mask else None,
                    labels=targets_input_ids[step_idx],
                    return_logits=should_log_debug  # Only return logits when needed
                )
            
            # Log current step loss and pred
            if should_log_debug:
                if loss_decode_t is None:
                    logger.debug(f"{mode_prefix} Loss Decode @ Step {step_idx}: skipped (no valid labels)")
                else:
                    val = loss_decode_t.item() if torch.isfinite(loss_decode_t) else float('nan')
                    logger.debug(f"{mode_prefix} Loss Decode @ Step {step_idx}: {val:.6f}")
                
                # Print target and pred
                if self.tokenizer:
                    target_ids_for_print = targets_input_ids[step_idx][0].clone()
                    target_ids_for_print[target_ids_for_print == -100] = self.tokenizer.pad_token_id
                    decoded_targ = self.tokenizer.decode(target_ids_for_print, skip_special_tokens=False)
                    
                    # Extract pred from logits
                    if logits_t is not None:
                        # Handle length mismatch
                        pred_ids_for_print = logits_t.argmax(dim=-1)[0]  # (seq_len, vocab_size) -> (seq_len,)
                        # Truncate or pad
                        target_len = target_ids_for_print.size(0)
                        if pred_ids_for_print.size(0) > target_len:
                            pred_ids_for_print = pred_ids_for_print[:target_len]
                        elif pred_ids_for_print.size(0) < target_len:
                            pad_len = target_len - pred_ids_for_print.size(0)
                            pred_ids_for_print = torch.cat([
                                pred_ids_for_print,
                                torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=pred_ids_for_print.dtype)
                            ])
                        decoded_pred = self.tokenizer.decode(pred_ids_for_print, skip_special_tokens=False)
                    else:
                        decoded_pred = "[logits not computed]"
                    
                    logger.debug(f"{mode_prefix} --- Decoding Step {step_idx} ---")
                    logger.debug(f"  Target: {decoded_targ}")
                    logger.debug(f"  Pred  : {decoded_pred}")
            
            # Accumulate loss
            weight = 1.0 / num_steps
            if loss_decode_t is not None and torch.isfinite(loss_decode_t):
                weighted_loss = weight * loss_decode_t
                if total_loss_reconstruct is None:
                    total_loss_reconstruct = weighted_loss
                else:
                    total_loss_reconstruct = total_loss_reconstruct + weighted_loss
                total_weight_reconstruct += weight
        
        # Compute final loss
        states = torch.stack(states_list, dim=1)
        del past_key_values
        torch.cuda.empty_cache()
        
        if total_weight_reconstruct > 0:
            loss_reconstruct = total_loss_reconstruct
        else:
            loss_reconstruct = torch.zeros(1, device=device, requires_grad=True).squeeze(0)
        
        total_loss = loss_reconstruct
        
        if should_log_debug:
            logger.debug(f"\n{mode_prefix} --- Final Losses ---")
            logger.debug(f"{mode_prefix} Loss Reconstruct: {loss_reconstruct.item():.6f}")
            logger.debug(f"{mode_prefix} Total Loss:      {total_loss.item():.6f}")
            logger.debug(f"{'='*50}\n")
        
        return PLaTOutput(
            loss=total_loss,
            loss_reconstruct=loss_reconstruct,
            logits=generated_logits if eval_mode else None,
            generated_trajectory=states
        )

    # =========================================================================
    #                             Inference Methods
    # =========================================================================
    
    def inference(self, question, max_steps=20, 
                  pass_at_k=1, temperature=1.0, do_sample=False,
                  skip_special_tokens=False, extract_answer=True, 
                  only_return_ids=False) -> Tuple[List[str], List[List[str]], Dict]:
        """
        Pure latent space reasoning
        
        Args:
            question: question text
            max_steps: max reasoning steps
            pass_at_k: number of samples per state
            temperature: sampling temperature
            
        Returns:
            final_answers: final answer list
            reasoning_steps_list: reasoning steps list
            token_stats: token statistics
        """
        self.eval()
        device = next(self.parameters()).device
        train_cfg = self.plat_config.train
        
        # Encode question
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'
        question_enc = self.tokenizer(
            question,
            max_length=128,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        self.tokenizer.padding_side = original_padding_side
        
        question_ids = question_enc['input_ids'].to(device)
        question_mask = question_enc['attention_mask'].to(device)
        
        question_embeds = self.backbone.get_input_embeddings()(question_ids)
        backbone_dtype = self._get_backbone_dtype()
        question_embeds = question_embeds.to(backbone_dtype)
        
        decoder_model = self.decoder_lm if self.use_independent_decoder else self.backbone
        only_ques_cache = decoder_model(
            inputs_embeds=question_embeds,
            attention_mask=question_mask,
            use_cache=True,
            return_dict=True
        ).past_key_values
        
        with torch.no_grad():
            # Encode initial state
            encode_token_tensor = torch.full((question_ids.size(0), 1), self.encode_token_id,
                                             dtype=question_ids.dtype, device=device)
            encode_attention_tensor = torch.ones((question_mask.size(0), 1),
                                                  dtype=question_mask.dtype, device=device)
            question_ids_enc = torch.cat([question_ids, encode_token_tensor], dim=1)
            question_mask_enc = torch.cat([question_mask, encode_attention_tensor], dim=1)
            s_0 = self.encode_state(question_ids_enc, question_mask_enc)
            
            # Generate state trajectory
            states_list = [s_0]
            past_key_values = None
            num_latent_per_step = self.num_latent_for_a_step
            slot_accumulators = [None] * num_latent_per_step
            slot_accumulators[0] = s_0
            
            ema_alpha = train_cfg.ema_alpha
            state_transition_mode = train_cfg.state_transition_mode
            add_dyn_ques_context = train_cfg.add_dyn_ques_context
            
            def _generate_next_state_infer():
                nonlocal past_key_values
                if len(states_list) > 1:
                    history = torch.stack(states_list[1:], dim=1)
                else:
                    history = torch.empty(1, 0, self.latent_dim, device=device)
                
                if self.planner_type == 'head':
                    s_t, past_key_values_inner = self.planner_forward_with_head(
                        question_embeds if add_dyn_ques_context else None,
                        s_0, history, past_key_values
                    )
                else:
                    s_t, past_key_values_inner = self.planner(
                        question_embeds if add_dyn_ques_context else None,
                        s_0, history, past_key_values=past_key_values
                    )
                
                past_key_values = past_key_values_inner
                states_list.append(s_t)
                return s_t
            
            for slot_idx in range(1, num_latent_per_step):
                s_extra = _generate_next_state_infer()
                slot_accumulators[slot_idx] = s_extra
            
            state_chunks_history = []
            for step_idx in range(max_steps):
                should_skip_gen = (step_idx == 0)
                
                if not should_skip_gen:
                    for slot_idx in range(num_latent_per_step):
                        s_new = _generate_next_state_infer()
                        if slot_accumulators[slot_idx] is None:
                            slot_accumulators[slot_idx] = s_new
                        else:
                            if state_transition_mode == 'ema':
                                slot_accumulators[slot_idx] = ema_alpha * s_new + (1 - ema_alpha) * slot_accumulators[slot_idx]
                            elif state_transition_mode == 'residual':
                                slot_accumulators[slot_idx] = slot_accumulators[slot_idx] + s_new
                            elif state_transition_mode == 'independent':
                                slot_accumulators[slot_idx] = s_new
                
                state_chunks = torch.stack(slot_accumulators, dim=1)
                state_chunks_history.append(state_chunks.clone())
            
            # Decode generated states
            reasoning_steps_list = [[] for _ in range(pass_at_k)]
            final_answers = [None] * pass_at_k
            stop_token_ids = [self.step_end_id, self.answer_end_id, self.tokenizer.eos_token_id]
            stop_token_ids = [tid for tid in stop_token_ids if tid is not None]
            
            for step_idx in range(len(state_chunks_history)):
                state_chunks = state_chunks_history[step_idx]
                if pass_at_k > 1:
                    state_chunks_expanded = state_chunks.expand(pass_at_k, -1, -1)
                else:
                    state_chunks_expanded = state_chunks
                
                # Prepare past states
                
                # Prepare prefix cache
                prefix_past_kv = None
                
                generated_ids_batch, _ = self.decode_state(
                    state_chunks=state_chunks_expanded,
                    prefix_past_kv=prefix_past_kv,
                    decode_mode='generate',
                    stop_token_ids=stop_token_ids,
                    max_length=80,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None
                )
                
                if only_return_ids:
                    continue
                
                # Process each sample result
                for i in range(pass_at_k):
                    if final_answers[i] is not None:
                        continue
                    
                    gen_ids = generated_ids_batch[i]
                    decoded_text = self.tokenizer.decode(gen_ids, skip_special_tokens=skip_special_tokens)
                    
                    # Check if contains answer
                    if ANSWER_END in decoded_text:
                        if extract_answer:
                            final_answers[i] = extract_answer_from_text(decoded_text)
                        else:
                            final_answers[i] = decoded_text
                    else:
                        reasoning_steps_list[i].append(decoded_text)
            
            # Use last step for no answer
            for i in range(pass_at_k):
                if final_answers[i] is None:
                    final_answers[i] = ""
        
        # Calculate token stats
        num_latent_per_step = self.num_latent_for_a_step
        
        # Calculate avg latent states
        total_latent_states = 0
        for step_idx in range(len(state_chunks_history)):
            num_states_for_step = (step_idx + 1) * num_latent_per_step  # decode_all_latents=True
            total_latent_states += num_states_for_step
        avg_latent_states = total_latent_states / len(state_chunks_history) if state_chunks_history else 0
        
        # Calculate avg peek tokens
        avg_peek_tokens = len(state_chunks_history)
        
        # Calculate avg answer tokens
        total_answer_tokens = 0
        answer_count = 0
        for ans in final_answers:
            if ans is not None and ans != "":
                total_answer_tokens += len(str(ans).split())
                answer_count += 1
        avg_answer_tokens = total_answer_tokens / answer_count if answer_count > 0 else 0
        
        token_stats = {
            'num_latent_states': avg_latent_states,
            'num_peek_tokens': avg_peek_tokens,
            'num_answer_tokens': avg_answer_tokens,
            'total_tokens': avg_latent_states + avg_peek_tokens + avg_answer_tokens
        }
        
        return final_answers, reasoning_steps_list, token_stats

    # =========================================================================
    #                         Rollout Methods (for RL training)
    # =========================================================================
    
    def generate_and_decode_rollout(self,
                                     question_input_ids,
                                     question_attention_mask,
                                     max_steps=20,
                                     temperature=1.0,
                                     do_sample=True,
                                     force_eval_mode=True,
                                     train_intermediate_steps=False):
        """
        Fused version: generate latent states while decoding
        For RL rollout, avoid generating unnecessary states
        
        Args:
            question_input_ids: (batch_size, seq_len)
            question_attention_mask: (batch_size, seq_len)
            max_steps: max reasoning steps
            temperature: sampling temperature
            do_sample: whether to sample
            force_eval_mode: force eval mode
            train_intermediate_steps: train intermediate steps
            
        Returns:
            final_generated_ids: List[Tensor]
            final_steps_counts: List[int]
            total_log_prob: Tensor
            states_list_batch: List[Tensor]
        """
        # Force eval mode
        if force_eval_mode:
            self.eval()
            if hasattr(self, 'backbone'):
                self.backbone.eval()
            if hasattr(self, 'encoder'):
                self.encoder.eval()
            if hasattr(self, 'decoder'):
                self.decoder.eval()
            if hasattr(self, 'planner'):
                self.planner.eval()
            if hasattr(self, 'decoder_lm'):
                self.decoder_lm.eval()
        
        batch_size = question_input_ids.size(0)
        device = question_input_ids.device
        
        # Step 1: Initialize
        # 1.1 Encode s_0
        encode_token_tensor = torch.full((batch_size, 1), self.encode_token_id, dtype=question_input_ids.dtype, device=device)
        encode_attention_tensor = torch.ones((batch_size, 1), dtype=question_attention_mask.dtype, device=device)
        question_input_ids_enc = torch.cat([question_input_ids, encode_token_tensor], dim=1)
        question_attention_mask_enc = torch.cat([question_attention_mask, encode_attention_tensor], dim=1)
        s_0 = self.encode_state(question_input_ids_enc, question_attention_mask_enc)
        
        # 1.2 Prepare question embeddings
        question_embeds = self.backbone.get_input_embeddings()(question_input_ids)
        backbone_dtype = self._get_backbone_dtype()
        question_embeds = question_embeds.to(backbone_dtype)
        
        # 1.4 Get special token ID
        stop_token_ids = [self.step_end_id, self.answer_end_id, self.tokenizer.eos_token_id]
        stop_token_ids = [tid for tid in stop_token_ids if tid is not None]
        
        # Step 2: Initialize state
        train_cfg = self.plat_config.train
        num_latent_per_step = self.num_latent_for_a_step
        state_transition_mode = train_cfg.state_transition_mode
        ema_alpha = train_cfg.ema_alpha
        
        # Use detach to avoid gradient accumulation
        states_list = [s_0.detach()]
        past_key_values = None
        
        def _generate_next_state():
            nonlocal past_key_values
            if len(states_list) > 1:
                history = torch.stack(states_list[1:], dim=1)
            else:
                history = torch.empty(batch_size, 0, self.latent_dim, device=device)
            
            if self.planner_type == 'head':
                s_t, past_key_values_inner = self.planner_forward_with_head(
                    question_embeds if train_cfg.add_dyn_ques_context else None,
                    s_0, history, past_key_values
                )
            else:
                s_t, past_key_values_inner = self.planner(
                    question_embeds if train_cfg.add_dyn_ques_context else None,
                    s_0, history, past_key_values=past_key_values
                )
            
            # Key: detach to avoid gradient
            if past_key_values_inner is not None and isinstance(past_key_values_inner, tuple):
                past_key_values_inner = tuple(
                    tuple(kv.detach() if kv is not None else None for kv in layer_past)
                    for layer_past in past_key_values_inner
                )
            past_key_values = past_key_values_inner
            
            # detach states
            states_list.append(s_t.detach())
            return s_t
        
        # Initialize slot accumulators
        slot_accumulators = [None] * num_latent_per_step
        slot_accumulators[0] = s_0.detach()
        
        # Generate initial extra latents
        if num_latent_per_step > 1:
            for slot_idx in range(1, num_latent_per_step):
                s_extra = _generate_next_state()
                slot_accumulators[slot_idx] = s_extra.detach()
        
        # Initialize cumulative sum
        if num_latent_per_step == 1:
            cumulative_sum = s_0.detach().clone()
        
        # Collect results
        accumulated_ids_list = [[] for _ in range(batch_size)]
        segment_lengths_list = [[] for _ in range(batch_size)]
        final_steps_counts = [0] * batch_size
        collected_log_probs = [0.0] * batch_size
        finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Step 3: Main loop - generate and decode
        for step_idx in range(max_steps):
            if finished_mask.all():
                break
            
            self.set_adapter_mode('planner')
            
            # 3.1 Planner: generate next state
            should_skip_gen = (step_idx == 0)
            
            if not should_skip_gen:
                if num_latent_per_step > 1:
                    for slot_idx in range(num_latent_per_step):
                        s_new = _generate_next_state()
                        if slot_accumulators[slot_idx] is None:
                            slot_accumulators[slot_idx] = s_new.detach()
                        else:
                            if state_transition_mode == 'ema':
                                slot_accumulators[slot_idx] = (ema_alpha * s_new + (1 - ema_alpha) * slot_accumulators[slot_idx]).detach()
                            elif state_transition_mode == 'residual':
                                slot_accumulators[slot_idx] = (slot_accumulators[slot_idx] + s_new.detach()).detach()
                            elif state_transition_mode == 'independent':
                                slot_accumulators[slot_idx] = s_new.detach()
                    
                    current_state_chunks = torch.stack(slot_accumulators, dim=1)
                else:
                    s_new = _generate_next_state()
                    
                    if state_transition_mode == 'ema':
                        states_tensor = torch.stack(states_list, dim=1)
                        cumulative_states = self._compute_ema_cumulative(states_tensor, alpha=ema_alpha)
                        s_generated = cumulative_states[:, -1, :].detach()
                    elif state_transition_mode == 'residual':
                        cumulative_sum = cumulative_sum + s_new.detach()
                        s_generated = cumulative_sum
                    elif state_transition_mode == 'independent':
                        s_generated = s_new.detach()
                    else:
                        cumulative_sum = cumulative_sum + s_new.detach()
                        s_generated = cumulative_sum
                    
                    current_state_chunks = s_generated.unsqueeze(1)
            else:
                if num_latent_per_step > 1:
                    current_state_chunks = torch.stack(slot_accumulators, dim=1)
                else:
                    current_state_chunks = cumulative_sum.unsqueeze(1)
            
            # 3.2 Decoder: decode for unfinished samples
            active_indices = (~finished_mask).nonzero().squeeze(-1)
            if active_indices.numel() == 0:
                break
            current_state_chunks_detached = current_state_chunks[active_indices].detach()
            self.set_adapter_mode('decoder')
            
            full_ids, full_logits = self.decode_state(
                state_chunks=current_state_chunks_detached,
                decode_mode='generate',
                stop_token_ids=stop_token_ids,
                max_length=80,
                do_sample=do_sample,
                temperature=temperature,
                return_logits=True
            )
            
            # Compute log_prob
            log_probs = F.log_softmax(full_logits, dim=-1)
            selected_log_probs = log_probs.gather(2, full_ids.unsqueeze(-1)).squeeze(-1)
            step_log_probs = selected_log_probs.sum(dim=1)
            
            # Check completion
            truly_finished_mask = (full_ids == self.answer_end_id).any(dim=1)
            newly_finished_indices = active_indices[truly_finished_mask]
            
            # Accumulate results
            for local_idx, global_idx in enumerate(active_indices.tolist()):
                collected_log_probs[global_idx] += step_log_probs[local_idx].item()
                
                curr_sample_ids = full_ids[local_idx]
                pad_mask = (curr_sample_ids == self.tokenizer.pad_token_id)
                if pad_mask.any():
                    actual_len = pad_mask.nonzero()[0, 0].item()
                else:
                    actual_len = curr_sample_ids.size(0)
                
                valid_ids = curr_sample_ids[:actual_len]
                accumulated_ids_list[global_idx].append(valid_ids)
                segment_lengths_list[global_idx].append(actual_len)
                final_steps_counts[global_idx] = step_idx + 1
            
            # Mark as finished
            if len(newly_finished_indices) > 0:
                finished_mask[newly_finished_indices] = True
            
            # Cleanup
            del full_ids, full_logits, log_probs, selected_log_probs, step_log_probs
        
        # Step 4: Handle unfinished samples
        # Already handled in main loop
        
        # Step 5: Return results
        final_generated_ids = []
        for i in range(batch_size):
            if accumulated_ids_list[i]:
                final_generated_ids.append(torch.cat(accumulated_ids_list[i], dim=0))
            else:
                final_generated_ids.append(torch.empty(0, dtype=torch.long, device=device))
        
        total_log_prob = torch.tensor(collected_log_probs, device=device)
        
        # Extract states_list
        states_list_batch = []
        for i in range(batch_size):
            sample_states = torch.stack([s[i].detach() for s in states_list], dim=0).cpu()
            states_list_batch.append(sample_states)
        
        return final_generated_ids, final_steps_counts, total_log_prob, states_list_batch, segment_lengths_list
    
    def _compute_ema_cumulative(self, states_tensor, alpha=0.1):
        """Compute EMA cumulative state"""
        batch_size, seq_len, hidden_dim = states_tensor.shape
        ema_states = torch.zeros_like(states_tensor)
        ema_states[:, 0, :] = states_tensor[:, 0, :]
        for t in range(1, seq_len):
            ema_states[:, t, :] = alpha * states_tensor[:, t, :] + (1 - alpha) * ema_states[:, t-1, :]
        return ema_states

    # =========================================================================
    #                         Lazy Inference
    # =========================================================================
    
    def inference_lazy(self, question, max_steps=20, 
                       pass_at_k=1, temperature=1.0, do_sample=False,
                       skip_special_tokens=False, extract_answer=True, only_return_ids=False):
        """
        Lazy inference: optimize decoding via peek mechanism
        
        Core optimization ideas:
        1. Remove redundant state stacking and tensor ops
        2. For each step, peek first token
        3. If peek is not answer_start, skip decode
        4. Only decode fully when peek is answer_start
        5. Delay string decode, only when necessary
        
        This significantly reduces decoding overhead.
        
        Args:
            question: question text
            max_steps: max reasoning steps
            pass_at_k: number of samples
            temperature: sampling temperature
            do_sample: whether to sample
            skip_special_tokens: skip special tokens
            extract_answer: extract answer number
            only_return_ids: only return token IDs
            
        Returns:
            final_answers: final answer list
            reasoning_steps_list: reasoning steps list
            token_stats: None
        """
        self.eval()
        device = self.device
        train_cfg = self.plat_config.train
        
        # Encode question
        question_max_len = self.plat_config.data.max_length
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'
        question_enc = self.tokenizer(
            question,
            max_length=question_max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        ).to(device)
        self.tokenizer.padding_side = original_padding_side
        
        question_ids = question_enc['input_ids']
        question_mask = question_enc['attention_mask']
        batch_size = question_ids.size(0)
        
        # question_embeds for planner
        question_embeds = self.backbone.get_input_embeddings()(question_ids)
        backbone_dtype = self._get_backbone_dtype()
        question_embeds = question_embeds.to(backbone_dtype)

        with torch.no_grad():
            # Encode initial state
            encode_token_tensor = torch.full((batch_size, 1), self.encode_token_id, dtype=torch.long, device=device)
            encode_attention_tensor = torch.ones((batch_size, 1), dtype=torch.long, device=device)
            question_ids_enc = torch.cat([question_ids, encode_token_tensor], dim=1)
            question_mask_enc = torch.cat([question_mask, encode_attention_tensor], dim=1)
            s_0 = self.encode_state(question_ids_enc, question_mask_enc)
            
            # Latent space autoregression
            reasoning_steps_list = [[] for _ in range(pass_at_k)]
            final_answers = [None] * pass_at_k
            stop_token_ids = [self.step_end_id, self.answer_end_id, self.tokenizer.eos_token_id]
            stop_token_ids = [tid for tid in stop_token_ids if tid is not None]
            
            # State generation
            states_list = [s_0]
            dyn_past_kv = None
            num_latent_per_step = self.num_latent_for_a_step
            slot_accumulators = [None] * num_latent_per_step
            slot_accumulators[0] = s_0
            ema_alpha = train_cfg.ema_alpha
            state_transition_mode = train_cfg.state_transition_mode
            add_dyn_ques = train_cfg.add_dyn_ques_context

            def _generate_next_state_optimized():
                nonlocal dyn_past_kv
                # Incremental generation
                history = states_list[-1].unsqueeze(1) if dyn_past_kv is not None else torch.empty(batch_size, 0, self.latent_dim, device=device)
                planner_ques = question_embeds if add_dyn_ques else None
                
                if self.planner_type == 'head':
                    s_t, dyn_past_kv = self.planner_forward_with_head(planner_ques, s_0, history, dyn_past_kv)
                else:
                    s_t, dyn_past_kv = self.planner(planner_ques, s_0, history, past_key_values=dyn_past_kv)
                
                states_list.append(s_t)
                return s_t

            # Initialize slot accumulators
            for slot_idx in range(1, num_latent_per_step):
                slot_accumulators[slot_idx] = _generate_next_state_optimized()

            found_answer_mask = [False] * pass_at_k
            
            for step_idx in range(max_steps):
                if all(found_answer_mask):
                    break
                
                # Generate new state
                if step_idx != 0:
                    for slot_idx in range(num_latent_per_step):
                        s_new = _generate_next_state_optimized()
                        if state_transition_mode == 'ema':
                            slot_accumulators[slot_idx] = ema_alpha * s_new + (1 - ema_alpha) * slot_accumulators[slot_idx]
                        elif state_transition_mode == 'residual':
                            slot_accumulators[slot_idx] = slot_accumulators[slot_idx] + s_new
                        else:
                            slot_accumulators[slot_idx] = s_new
                
                state_chunks = torch.stack(slot_accumulators, dim=1)
                
                # Prepare decode prefix
                prefix_past_kv = None
                
                state_chunks_expanded = state_chunks.expand(pass_at_k, -1, -1) if pass_at_k > 1 else state_chunks
                
                # Peek (decode only 1 token)
                peek_ids, _ = self.decode_state(
                    state_chunks=state_chunks_expanded,
                    prefix_past_kv=prefix_past_kv,
                    decode_mode='generate',
                    stop_token_ids=stop_token_ids,
                    max_length=1,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None
                )
                
                is_answer_start = (peek_ids == self.answer_start_id).any(dim=1) if pass_at_k > 1 else (peek_ids == self.answer_start_id).any()
                
                # Dispatch decode tasks
                need_full_decode = [not found_answer_mask[i] and (is_answer_start[i] if pass_at_k > 1 else is_answer_start) for i in range(pass_at_k)]
                
                if any(need_full_decode):
                    decode_indices = [i for i, val in enumerate(need_full_decode) if val]
                    if pass_at_k > 1:
                        d_state_chunks = state_chunks_expanded[decode_indices]
                        d_prefix_kv = None
                        if prefix_past_kv is not None:
                            d_prefix_kv = tuple(tuple(kv[decode_indices] for kv in layer_past) for layer_past in prefix_past_kv)
                    else:
                        d_state_chunks, d_prefix_kv = state_chunks_expanded, prefix_past_kv
                    
                    full_ids, _ = self.decode_state(
                        state_chunks=d_state_chunks,
                        prefix_past_kv=d_prefix_kv,
                        decode_mode='generate',
                        stop_token_ids=stop_token_ids,
                        max_length=80,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else None
                    )
                    
                    generated_ids_batch = [None] * pass_at_k
                    if pass_at_k > 1:
                        for local_idx, global_idx in enumerate(decode_indices):
                            generated_ids_batch[global_idx] = full_ids[local_idx]
                    else:
                        generated_ids_batch = [full_ids[0]]
                else:
                    generated_ids_batch = [peek_ids[i] for i in range(pass_at_k)] if pass_at_k > 1 else [peek_ids[0]]
                
                # Collect results
                for i in range(pass_at_k):
                    if found_answer_mask[i] or generated_ids_batch[i] is None:
                        continue
                    is_ans = generated_ids_batch[i][0] == self.answer_start_id
                    if is_ans and extract_answer:
                        text = self.tokenizer.decode(generated_ids_batch[i], skip_special_tokens=False)
                        clean = text.replace(STEP_START, '').replace(STEP_END, '').replace(ANSWER_START, '').replace(ANSWER_END, '').replace('<|endoftext|>', '').strip() if skip_special_tokens else text
                        final_answers[i] = extract_answer_from_text(clean)
                        found_answer_mask[i] = True
                    else:
                        text = self.tokenizer.decode(generated_ids_batch[i], skip_special_tokens=False)
                        clean = text.replace(STEP_START, '').replace(STEP_END, '').replace(ANSWER_START, '').replace(ANSWER_END, '').replace('<|endoftext|>', '').strip() if skip_special_tokens else text
                        reasoning_steps_list[i].append(clean)
            
            if only_return_ids:
                return generated_ids_batch
        
        return final_answers, reasoning_steps_list, None

    @classmethod
    def from_plat_config(cls, plat_config: PLaTConfig):
        """Create model from PLaTConfig"""
        base_name = plat_config.model.sft_model_path or plat_config.model.base_model_name
        base_config = PretrainedConfig.from_pretrained(base_name)
        return cls(base_config, plat_config)
    
    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str, base_model_path: str = None):
        """
        Load model from incremental checkpoint
        
        Args:
            checkpoint_dir: checkpoint directory
                - plat_config.yaml
                - lora_adapter/
                - plat_modules.bin
                - training_metadata.json
            base_model_path: base model path
            
        Returns:
            PLaTModel: loaded model
        """
        import os
        import json
        
        # Load config
        config_path = os.path.join(checkpoint_dir, 'plat_config.yaml')
        if os.path.exists(config_path):
            plat_config = PLaTConfig.load(config_path)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load metadata
        metadata_path = os.path.join(checkpoint_dir, 'training_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Determine base model path
        if base_model_path is None:
            base_model_path = metadata.get('base_model_path') or plat_config.model.sft_model_path or plat_config.model.base_model_name
        
        plat_config.model.sft_model_path = base_model_path
        
        # Create model:
        # Avoid double wrapping
        lora_enabled = getattr(plat_config.lora, "enabled", False)
        if lora_enabled:
            plat_config.lora.enabled = False
        model = cls.from_plat_config(plat_config)
        if lora_enabled:
            plat_config.lora.enabled = True
        
        # Load LoRA adapter
        lora_dir = os.path.join(checkpoint_dir, 'lora_adapter')
        if os.path.exists(lora_dir) and metadata.get('has_lora', False):
            # Align vocab size
            try:
                from transformers import AutoTokenizer
                ckpt_tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
                target_vocab_size = len(ckpt_tokenizer)
                current_vocab_size = model.backbone.get_input_embeddings().weight.size(0)
                if current_vocab_size != target_vocab_size:
                    model.backbone.resize_token_embeddings(target_vocab_size, mean_resizing=False)
                    model.backbone.config.vocab_size = target_vocab_size
                    logger.info(f"Align vocab size before loading LoRA: {current_vocab_size} -> {target_vocab_size}")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}")
            model.backbone = PeftModel.from_pretrained(model.backbone, lora_dir)
            logger.info(f"Loaded LoRA adapter: {lora_dir}")
        
        # Load PLaT modules
        plat_path = os.path.join(checkpoint_dir, 'plat_modules.bin')
        if os.path.exists(plat_path):
            plat_state_dict = torch.load(plat_path, map_location='cpu')
            
            # Load into model
            missing_keys = []
            for name, param in plat_state_dict.items():
                try:
                    # Get model params
                    parts = name.split('.')
                    module = model
                    for part in parts[:-1]:
                        module = getattr(module, part)
                    
                    target_param = getattr(module, parts[-1])
                    if isinstance(target_param, nn.Parameter):
                        target_param.data.copy_(param)
                    else:
                        setattr(module, parts[-1], nn.Parameter(param))
                except Exception as e:
                    missing_keys.append(name)
                    logger.warning(f"Cannot load parameter {name}: {e}")
            
            if missing_keys:
                logger.warning(f"Failed to load params: {missing_keys}")
            
            logger.info(f"Loaded PLaT modules: {plat_path}")
        
        # Try loading full model
        full_model_path = os.path.join(checkpoint_dir, 'pytorch_model.bin')
        if os.path.exists(full_model_path) and not os.path.exists(plat_path):
            state_dict = torch.load(full_model_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded full model: {full_model_path}")
        
        logger.info(f"Model loaded from checkpoint: {checkpoint_dir}")
        return model
    
    def get_trainable_parameters_info(self):
        """Get trainable parameter info for logging"""
        total_params = 0
        trainable_params = 0
        trainable_groups = {}
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
                # Group stats
                group = name.split('.')[0]
                if group not in trainable_groups:
                    trainable_groups[group] = 0
                trainable_groups[group] += param.numel()
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'trainable_ratio': trainable_params / total_params if total_params > 0 else 0,
            'trainable_groups': trainable_groups
        }