#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import shutil
from datetime import datetime
import argparse
import torch
import logging
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    PretrainedConfig,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint

# 
sys.path.insert(0, '/home/jc/workspace/inf')

from plat.config.base_config import (
    PLaTConfig, SPECIAL_TOKENS,
    STEP_START, STEP_END, ANSWER_START, ANSWER_END,
    ENCODE_TOKEN, DECODE_TOKEN, PLANNER_TOKEN
)
from plat.data.reasoning_dataset import (
    PLaTDataset, get_gsm8k_dataset, plat_collate_fn
)
from plat.models.plat_model import PLaTModel
from plat.engine.trainer import PLaTTrainer


def setup_logging(output_dir: str, is_main_process: bool):
    """，print"""
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        log_filename = os.path.join(output_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # handler
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # handler：DEBUG
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # handler：INFO（cmd）
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        logger = logging.getLogger(__name__)
        logger.info(f": {log_filename}")
        logger.info("INFO，DEBUG")
        
        # printlogger
        import builtins
        _original_print = builtins.print
        
        def new_print(*args, **kwargs):
            message = ' '.join(str(arg) for arg in args)
            logger.info(message)
        
        builtins.print = new_print
        
        return logger
    else:
        # ：
        root_logger = logging.getLogger()
        root_logger.handlers = []
        root_logger.addHandler(logging.NullHandler())
        
        # logger
        for logger_name in ['transformers', 'transformers.trainer', 'transformers.configuration_utils',
                           'transformers.modeling_utils', 'transformers.tokenization_utils_base',
                           'torch', 'torch.distributed', 'deepspeed']:
            lib_logger = logging.getLogger(logger_name)
            lib_logger.handlers = []
            lib_logger.addHandler(logging.NullHandler())
            lib_logger.propagate = False
        
        # logger
        logger = logging.getLogger(__name__)
        logger.handlers = []
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        
        # print
        import builtins
        _original_print = builtins.print
        
        def new_print(*args, **kwargs):
            pass  # 
        
        builtins.print = new_print
        
        return logger


def check_main_process():
    """"""
    rank = os.environ.get('RANK', None)
    local_rank_env = os.environ.get('LOCAL_RANK', None)
    
    if rank is not None:
        return int(rank) == 0
    elif local_rank_env is not None:
        return int(local_rank_env) == 0
    else:
        return True


def main():
    parser = argparse.ArgumentParser(description="PLaT")
    parser.add_argument("--config", type=str, default=None, help="（YAML）")
    parser.add_argument("--resume_from", type=str, default=None, help="checkpoint")
    parser.add_argument("--sft_model_path", type=str, default=None, help="SFT")
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    args = parser.parse_args()
    
    is_main_process = check_main_process()
    
    # 
    if args.config:
        config = PLaTConfig.load(args.config)
    else:
        config = PLaTConfig()
    
    # 
    if args.sft_model_path:
        config.model.sft_model_path = args.sft_model_path
    
    # 
    planner_type = config.model.planner.type
    run_name = f"plat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{planner_type}"
    output_dir = os.path.join(config.train.model_save_dir, run_name)
    
    #  config.yaml （）
    os.makedirs(output_dir, exist_ok=True)
    config.save(os.path.join(output_dir, "config.yaml"))
    
    # （ checkpoint ）
    scripts_to_save = [
        '/home/jc/workspace/inf/plat/scripts/train_plat.py',
        '/home/jc/workspace/inf/plat/models/plat_model.py',
        '/home/jc/workspace/inf/plat/models/cot_model.py',
        '/home/jc/workspace/inf/plat/models/base_reasoning_model.py',
    ]

    # 
    logger = setup_logging(output_dir, is_main_process)

    for src_path in scripts_to_save:
        if os.path.exists(src_path):
            dst_path = os.path.join(output_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dst_path)
            if is_main_process:
                logger.info(f" {os.path.basename(src_path)}  checkpoint")

    if is_main_process:
        logger.info(f" config.yaml : {output_dir}")
    
    
    if is_main_process:
        logger.info(f"PLaT Training Script")
        logger.info(f"Run name: {run_name}")
        logger.info(f"Output dir: {output_dir}")
    
    # 
    resume_from_checkpoint = args.resume_from
    resumed_metadata = None
    
    if resume_from_checkpoint and os.path.isdir(resume_from_checkpoint):
        import json
        metadata_path = os.path.join(resume_from_checkpoint, 'training_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                resumed_metadata = json.load(f)
            if is_main_process:
                logger.info(f"checkpoint: {resumed_metadata}")
    
    # 
    if resumed_metadata and resumed_metadata.get('run_name'):
        run_name = resumed_metadata['run_name']
        output_dir = os.path.join(config.train.model_save_dir, run_name)
        if is_main_process:
            logger.info(f": {run_name}")
    
    # Wandb
    if config.wandb.enabled and is_main_process:
        import wandb
        os.environ["WANDB_PROJECT"] = config.wandb.project
        
        if resumed_metadata and resumed_metadata.get('wandb_run_id'):
            wandb.init(
                project=config.wandb.project,
                name=run_name,
                id=resumed_metadata['wandb_run_id'],
                resume="must",
                config=config.to_dict()
            )
        else:
            wandb.init(
                project=config.wandb.project,
                name=run_name,
                config=config.to_dict()
            )
    elif config.wandb.enabled and not is_main_process:
        os.environ["WANDB_MODE"] = "disabled"
    
    # SFT
    sft_model_path = config.model.sft_model_path or config.model.base_model_name
    if not os.path.exists(sft_model_path):
        raise ValueError(f": {sft_model_path}")
    
    if is_main_process:
        logger.info(f"Using model from: {sft_model_path}")
    
    # tokenizer
    tokenizer_path = sft_model_path
    if resume_from_checkpoint and os.path.isdir(resume_from_checkpoint):
        # checkpointtokenizer
        checkpoint_tokenizer_files = ['tokenizer_config.json', 'vocab.json', 'merges.txt']
        has_tokenizer = any(os.path.exists(os.path.join(resume_from_checkpoint, f)) 
                          for f in checkpoint_tokenizer_files)
        if has_tokenizer:
            tokenizer_path = resume_from_checkpoint
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if is_main_process:
        logger.info(f"Loaded tokenizer with vocab size: {len(tokenizer)}")
    
    #  pad token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if is_main_process:
            logger.info(f"Added [PAD] token. New vocab size: {len(tokenizer)}")
    
    # token
    missing_tokens = []
    for token in SPECIAL_TOKENS['additional_special_tokens']:
        if tokenizer.convert_tokens_to_ids(token) == tokenizer.unk_token_id:
            missing_tokens.append(token)
    
    if missing_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': missing_tokens})
        if is_main_process:
            logger.info(f"Added missing tokens: {missing_tokens}. New vocab size: {len(tokenizer)}")
    
    # vocab_size
    config.model.vocab_size = len(tokenizer)
    config.model.pad_token_id = tokenizer.pad_token_id
    
    # 
    if is_main_process:
        logger.info("Loading dataset...")
    
    raw_dataset = get_gsm8k_dataset(
        data_dir=config.data.data_dir,
        cache_dir=os.path.join(config.data.data_dir, "cache")
    )
    
    # 
    if config.data.debug_mode:
        if is_main_process:
            logger.info(f"!!! DEBUG MODE: Using limited samples !!!")
        raw_dataset['train'] = raw_dataset['train'].select(
            range(min(config.data.debug_train_samples, len(raw_dataset['train'])))
        )
        raw_dataset['valid'] = raw_dataset['valid'].select(
            range(min(config.data.debug_eval_samples, len(raw_dataset['valid'])))
        )
    
    # 
    train_dataset = PLaTDataset(raw_dataset['train'], tokenizer, max_length=config.data.max_length)
    eval_dataset = PLaTDataset(raw_dataset['valid'], tokenizer, max_length=config.data.max_length)
    
    if is_main_process:
        logger.info(f"Train dataset: {len(train_dataset)} samples")
        logger.info(f"Eval dataset: {len(eval_dataset)} samples")
    
    # 
    if is_main_process:
        logger.info("Initializing PLaT model...")
    
    base_config = PretrainedConfig.from_pretrained(sft_model_path)
    base_config.vocab_size = len(tokenizer)
    
    model = PLaTModel(base_config, config)
    model.set_tokenizer(tokenizer)
    
    #  backbone config  model_type（ checkpoint）
    sft_model_path = config.model.sft_model_path or config.model.base_model_name
    if not hasattr(model.backbone.config, 'model_type') or model.backbone.config.model_type is None:
        try:
            from transformers import AutoConfig
            base_config_check = AutoConfig.from_pretrained(sft_model_path)
            if hasattr(base_config_check, 'model_type') and base_config_check.model_type:
                model.backbone.config.model_type = base_config_check.model_type
                if is_main_process:
                    logger.info(f" model_type: {base_config_check.model_type}")
            else:
                raise ValueError(f" {sft_model_path}  model_type")
        except Exception as e:
            if is_main_process:
                logger.error(f" model_type : {e}")
            raise e

    #  pad_token_id， transformers  eos_token_id 
    model.backbone.config.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
    
    # tokenembedding
    if missing_tokens:
        with torch.no_grad():
            source_token_id = tokenizer.convert_tokens_to_ids('.')
            embeddings = model.backbone.get_input_embeddings()
            
            for token_name in missing_tokens:
                token_id = tokenizer.convert_tokens_to_ids(token_name)
                if token_id >= embeddings.weight.size(0) - len(missing_tokens):
                    embeddings.weight[token_id] = embeddings.weight[source_token_id].clone()
                    if is_main_process:
                        logger.info(f"Initialized embedding for '{token_name}' (ID: {token_id})")
    
    if is_main_process:
        # param names
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"Trainable: {name} (shape: {param.shape})")
        # statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # TrainingArguments
    train_cfg = config.train
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=train_cfg.learning_rate,
        num_train_epochs=train_cfg.num_train_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        warmup_ratio=train_cfg.warmup_ratio,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        weight_decay=train_cfg.weight_decay,
        eval_strategy=train_cfg.eval_strategy,
        eval_steps=train_cfg.eval_steps,
        save_strategy=train_cfg.save_strategy,
        save_steps=train_cfg.save_steps,
        save_total_limit=train_cfg.save_total_limit,
        load_best_model_at_end=train_cfg.load_best_model_at_end,
        metric_for_best_model=train_cfg.metric_for_best_model,
        logging_steps=train_cfg.logging_steps,
        bf16=train_cfg.bf16,
        fp16=train_cfg.fp16,
        deepspeed=train_cfg.deepspeed,
        dataloader_num_workers=train_cfg.dataloader_num_workers,
        seed=train_cfg.seed,
        report_to="wandb" if config.wandb.enabled else "none",
    )
    
    # callbacks
    callbacks = []
    if train_cfg.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=train_cfg.early_stopping_patience,
            early_stopping_threshold=train_cfg.early_stopping_threshold
        ))
    
    # Trainer
    trainer = PLaTTrainer(
        plat_config=config,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=plat_collate_fn,
        callbacks=callbacks if callbacks else None,
        backbone_learning_rate=train_cfg.backbone_learning_rate,
        freeze_backbone=train_cfg.freeze_backbone,
        tokenizer_to_save=tokenizer,
        run_name=run_name,
    )
    
    # 
    if is_main_process:
        logger.info("\n" + "=" * 80)
        logger.info("Starting PLaT Training")
        logger.info("=" * 80)
    
    # checkpoint
    if not resume_from_checkpoint and os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint:
            if is_main_process:
                logger.info(f"output_dircheckpoint: {last_checkpoint}")
            resume_from_checkpoint = last_checkpoint
            # checkpointrun_name
            import json
            metadata_path = os.path.join(resume_from_checkpoint, 'training_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    checkpoint_metadata = json.load(f)
                if checkpoint_metadata and checkpoint_metadata.get('run_name'):
                    run_name = checkpoint_metadata['run_name']
                    output_dir = os.path.join(config.train.model_save_dir, run_name)
                    if is_main_process:
                        logger.info(f"checkpoint: {run_name}")
    
    if resume_from_checkpoint and is_main_process:
        logger.info(f"checkpoint: {resume_from_checkpoint}")
    elif is_main_process:
        logger.info("")
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    if is_main_process:
        logger.info("=" * 80)
        logger.info("Training completed!")
        logger.info("=" * 80)
    
    # 
    final_model_path = os.path.join(output_dir, "final_checkpoint")

    #  config  model_type
    if not hasattr(model.backbone.config, 'model_type') or model.backbone.config.model_type is None:
        try:
            from transformers import AutoConfig
            base_config_final = AutoConfig.from_pretrained(sft_model_path)
            if hasattr(base_config_final, 'model_type') and base_config_final.model_type:
                model.backbone.config.model_type = base_config_final.model_type
                if is_main_process:
                    logger.info(f" model_type: {base_config_final.model_type}")
            else:
                raise ValueError(f" {sft_model_path}  model_type")
        except Exception as e:
            if is_main_process:
                logger.error(f" model_type : {e}")
            raise e

    #  pad_token_id， transformers  eos_token_id 
    model.backbone.config.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    # PLaT
    config.save(os.path.join(final_model_path, "config.yaml"))
    
    if is_main_process:
        logger.info(f"Final PLaT model saved to: {final_model_path}")
    
    if config.wandb.enabled and is_main_process:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
