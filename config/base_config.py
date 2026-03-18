# -*- coding: utf-8 -*-
"""
PLaT Framework Configuration
1. Components: atomic configs
2. Base: training logic base class
3. Method-Specific: CoT SFT and PLaT configs
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import yaml

# =================================================================================
# 1. Components
# =================================================================================

@dataclass
class DataConfig:
    data_dir: str = "./data"
    max_length: int = 128
    debug_mode: bool = False
    debug_train_samples: int = 500
    debug_eval_samples: int = 50

@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "plat"
    experiment: str = "default"

@dataclass
class LoRAConfig:
    enabled: bool = True
    r: int = 128
    lora_alpha: int = 32
    lora_dropout: float = 0.0

@dataclass
class LatentDenoisingConfig:
    """Latent Denoising Config"""
    enabled: bool = True
    noise_std: float = 0.1
    min_noise_std: float = 0.01
    start_epoch: int = 0
    end_epoch: int = 10
    noise_schedule: str = 'constant'  # 'constant', 'linear_decay', 'exponential_decay'
    apply_to_history_only: bool = True

# =================================================================================
# 2. Base (common training logic)
# =================================================================================

@dataclass
class BaseTrainConfig:
    learning_rate: float = 5e-4
    num_train_epochs: int = 25
    per_device_train_batch_size: int = 128
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.0
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 500
    eval_strategy: str = "steps"
    eval_steps: int = 500
    early_stopping_patience: int = 10000
    early_stopping_threshold: float = 0.001
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_acc"
    greater_is_better: bool = True
    bf16: bool = True
    fp16: bool = False
    seed: int = 42
    model_save_dir: str = "./outputs"
    dataloader_num_workers: int = 16
    preprocessing_num_workers: int = 8
    freeze_backbone: bool = True

    deepspeed: str = None

# =================================================================================
# 3. Method-specific config containers
# =================================================================================

class ConfigBase:
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    def save(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

# --- CoT SFT Method Config ---
@dataclass
class CoTSFTConfig(ConfigBase):
    base_model_name: str = "gpt2"
    sft_model_path: Optional[str] = None
    vocab_size: Optional[int] = None
    
    data: DataConfig = field(default_factory=DataConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    train: BaseTrainConfig = field(default_factory=BaseTrainConfig)

    @classmethod
    def load(cls, path: str) -> 'CoTSFTConfig':
        with open(path, 'r') as f:
            d = yaml.safe_load(f) or {}
        
        # Ensure numeric types are correct
        train_data = d.get('train', {})
        if 'learning_rate' in train_data and isinstance(train_data['learning_rate'], str):
            train_data['learning_rate'] = float(train_data['learning_rate'])
        
        return cls(
            base_model_name=d.get('base_model_name', "gpt2"),
            sft_model_path=d.get('sft_model_path'),
            vocab_size=d.get('vocab_size'),
            data=DataConfig(**d.get('data', {})),
            wandb=WandbConfig(**d.get('wandb', {})),
            lora=LoRAConfig(**d.get('lora', {})),
            train=BaseTrainConfig(**train_data)
        )

# --- PLaT Method Config ---
@dataclass
class PlannerConfig:
    """PLaT Planner Model Config"""
    type: str = "backbone_based"
    num_layers: int = 4
    num_heads: int = 8
    hidden_dim: int = 2048
    add_extra_layers: int = 2
    use_top_n_layers: int = -1

@dataclass
class EncoderConfig:
    """PLaT Encoder Config"""
    input_dim: int = 768
    hidden_dim: int = 2048
    num_layers: int = 2
    activation: str = "relu"

@dataclass
class DecoderConfig:
    """PLaT Decoder Config"""
    latent_dim: int = 2048
    output_dim: int = 768

@dataclass
class PLaTTrainConfig(BaseTrainConfig):
    """PLaT Train Config"""
    backbone_learning_rate: float = 0.0001

    ema_alpha: float = 0.5
    state_transition_mode: str = 'ema'
    add_dyn_ques_context: bool = True
    latent_denoising: LatentDenoisingConfig = field(default_factory=LatentDenoisingConfig)

@dataclass
class RLConfig:
    """RL Train Config"""
    # Generation related
    max_steps: int = 20
    temperature: float = 0.9
    do_sample: bool = True
    num_generations: int = 4
    
    # KL penalty
    kl_beta: float = 0.1
    clip_eps: float = 0.2
    
    # Reward config
    correct_answer_reward: float = 1.0
    incorrect_answer_penalty: float = 0.0
    valid_answer_reward: float = 0.1
    invalid_answer_penalty: float = 0.1
    valid_equation_reward: float = 0.1
    
    # Train intermediate steps
    train_intermediate_steps: bool = True

@dataclass
class PLaTModelConfig:
    """PLaT Model Config"""
    base_model_name: str = "gpt2"
    sft_model_path: str = None
    latent_dim: int = 2048
    vocab_size: Optional[int] = None
    num_latent_for_a_step: int = 2
    use_independent_decoder: bool = False
        
    # Component configs
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

@dataclass
class PLaTConfig(ConfigBase):
    """PLaT Full Config"""
    model: PLaTModelConfig = field(default_factory=PLaTModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    train: PLaTTrainConfig = field(default_factory=PLaTTrainConfig)
    rl: RLConfig = field(default_factory=RLConfig)

    @classmethod
    def load(cls, path: str) -> 'PLaTConfig':
        with open(path, 'r') as f:
            d = yaml.safe_load(f) or {}
        m = d.get('model', {})
        
        # Extract train config
        train_data = d.get('train', {})
        
        # Extract denoising config
        denoising_cfg_dict = train_data.get('latent_denoising', {})
        denoising_cfg = LatentDenoisingConfig(**denoising_cfg_dict)
        
        # Extract RL config
        rl_cfg_dict = d.get('rl', {})
        rl_cfg = RLConfig(**rl_cfg_dict)
        
        # Build PLaTTrainConfig
        plat_train_cfg = PLaTTrainConfig(
            **{k: v for k, v in train_data.items() if k != 'latent_denoising'},
            latent_denoising=denoising_cfg
        )
        
        return cls(
            model=PLaTModelConfig(
                base_model_name=m.get('base_model_name', "gpt2"),
                sft_model_path=m.get('sft_model_path'),
                latent_dim=m.get('latent_dim', 2048),
                vocab_size=m.get('vocab_size'),
                num_latent_for_a_step=m.get('num_latent_for_a_step', 2),
                use_independent_decoder=m.get('use_independent_decoder', False),
                planner=PlannerConfig(**m.get('planner', {})),
                encoder=EncoderConfig(**m.get('encoder', {})),
                decoder=DecoderConfig(**m.get('decoder', {}))
            ),
            data=DataConfig(**d.get('data', {})),
            wandb=WandbConfig(**d.get('wandb', {})),
            lora=LoRAConfig(**d.get('lora', {})),
            train=plat_train_cfg,
            rl=rl_cfg,
        )

# =================================================================================
# 4. Constants
# =================================================================================
STEP_START, STEP_END = '<STEP_START>', '<STEP_END>'
ANSWER_START, ANSWER_END = '<ANSWER_START>', '<ANSWER_END>'
ENCODE_TOKEN, DECODE_TOKEN = '<ENCODE>', '<DECODE>'
PLANNER_TOKEN = '<PLAN>'

SPECIAL_TOKENS = {
    'additional_special_tokens': [
        STEP_START, STEP_END, ANSWER_START, ANSWER_END,
        ENCODE_TOKEN, DECODE_TOKEN, PLANNER_TOKEN
    ]
}
