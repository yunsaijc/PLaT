# -*- coding: utf-8 -*-
"""
PLaTTrainer

：
1. TrainingState: 
2. PLaTTrainer: HuggingFace Trainer
"""

from dataclasses import dataclass
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List
import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import logging

from plat.config.base_config import PLaTConfig, PLaTTrainConfig
from plat.data.reasoning_dataset import StepGroupedBatchSampler, plat_collate_fn
from plat.utils.helpers import extract_answer_from_text, compare_answers

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """
    ，
    """
    current_epoch: float = 0.0
    global_step: int = 0
    is_eval: bool = False
    total_epochs: int = 1


class PLaTTrainer(Trainer):
    """
    PLaTTrainer
    
    ：
    1. StepGroupedBatchSampler
    2. 
    3. 
    4. 
    """
    def __init__(self, 
                 plat_config: PLaTConfig,
                 *args, 
                 backbone_learning_rate: float = None,
                 freeze_backbone: bool = True,
                 tokenizer_to_save=None,
                 run_name: str = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.plat_config = plat_config
        self.backbone_learning_rate = backbone_learning_rate
        self.freeze_backbone = freeze_backbone
        self.tokenizer_to_save = tokenizer_to_save
        self.run_name = run_name
        
        # 
        self.current_epoch = 0
        self.training_state = TrainingState()
        
        # 
        self._last_loss_reconstruct = []
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """epochsamplerepoch"""
        super().on_epoch_begin(args, state, control, **kwargs)
        self.current_epoch = int(state.epoch)
        
        # samplerepoch
        train_dataloader = self.get_train_dataloader()
        if hasattr(train_dataloader, 'batch_sampler') and \
           isinstance(train_dataloader.batch_sampler, StepGroupedBatchSampler):
            train_dataloader.batch_sampler.set_epoch(int(state.epoch))
    
    def on_step_end(self, args, state, control, **kwargs):
        """stepcurrent_epoch"""
        super().on_step_end(args, state, control, **kwargs)
        self.current_epoch = state.epoch
        return control
    
    def get_train_dataloader(self) -> DataLoader:
        """，StepGroupedBatchSampler"""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        batch_sampler = StepGroupedBatchSampler(
            dataset=self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            drop_last=self.args.dataloader_drop_last,
        )
        
        dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        
        if isinstance(dataloader.batch_sampler, StepGroupedBatchSampler):
            dataloader.batch_sampler.set_epoch(self.current_epoch)
        
        return dataloader
    
    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """，StepGroupedBatchSampler"""
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
        batch_sampler = StepGroupedBatchSampler(
            dataset=eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            drop_last=False,
        )
        
        return DataLoader(
            eval_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """，TrainingState"""
        # 
        self.training_state.current_epoch = self.current_epoch
        self.training_state.global_step = self.state.global_step
        self.training_state.is_eval = not model.training
        self.training_state.total_epochs = int(self.args.num_train_epochs)
        
        outputs = model(
            question_input_ids=inputs['question_input_ids'],
            question_attention_mask=inputs['question_attention_mask'],
            targets_input_ids=inputs['targets_input_ids'],
            targets_attention_mask=inputs.get('targets_attention_mask'),
            training_state=self.training_state,
            eval_mode=False,
        )
        
        loss = outputs.loss
        
        # 
        if outputs.loss_reconstruct is not None:
            self._last_loss_reconstruct.append(outputs.loss_reconstruct.mean().item())
        
        return (loss, outputs) if return_outputs else loss
    
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, 
                                 ignore_keys_for_eval, start_time, learning_rate=None):
        """"""
        self.current_epoch = int(self.state.epoch)
        outputs = super()._maybe_log_save_evaluate(
            tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate
        )
        
        # logging_steps
        if self.state.global_step % self.args.logging_steps == 0:
            if len(self._last_loss_reconstruct) > 0:
                extra_logs = {
                    "loss_reconstruct": float(np.mean(self._last_loss_reconstruct)),
                }
                
                self.log(extra_logs)
                
                # 
                self._last_loss_reconstruct = []
        
        return outputs
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, 
                        ignore_keys=None, metric_key_prefix="eval"):
        """evaluation_loop，"""
        # 
        if hasattr(self, '_eval_correct_count'):
            del self._eval_correct_count
        if hasattr(self, '_eval_num_samples'):
            del self._eval_num_samples
        if hasattr(self, '_eval_loss_decode_sum'):
            del self._eval_loss_decode_sum
        if hasattr(self, '_eval_num_batches'):
            del self._eval_num_batches
        
        output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )
        
        # 
        if hasattr(self, '_eval_loss_decode_sum'):
            num_batches = self._eval_num_batches if hasattr(self, '_eval_num_batches') and self._eval_num_batches > 0 else 1
            avg_loss_decode = self._eval_loss_decode_sum / num_batches
            output.metrics[f'{metric_key_prefix}_loss_decode'] = avg_loss_decode
            del self._eval_loss_decode_sum
            del self._eval_num_batches
        
        # 
        #  eval_acc （0）
        if not hasattr(self, '_eval_correct_count'):
            self._eval_correct_count = 0
        if not hasattr(self, '_eval_num_samples'):
            self._eval_num_samples = 0
        
        if hasattr(self, '_eval_num_samples') and self._eval_num_samples > 0:
            accuracy = self._eval_correct_count / self._eval_num_samples
        else:
            accuracy = 0.0
        
        output.metrics[f'{metric_key_prefix}_acc'] = accuracy
        logger.debug(f"[EVAL] eval_acc = {accuracy} ({self._eval_correct_count}/{self._eval_num_samples})")
        
        # 
        if hasattr(self, '_eval_correct_count'):
            del self._eval_correct_count
        if hasattr(self, '_eval_num_samples'):
            del self._eval_num_samples
        
        return output
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """prediction_step"""
        has_labels = "targets_input_ids" in inputs
        
        model.eval()
        device = next(model.parameters()).device
        
        # 
        inputs_on_device = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs_on_device[key] = value.to(device)
            elif isinstance(value, list):
                inputs_on_device[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
            else:
                inputs_on_device[key] = value
        
        with torch.no_grad():
            # 
            self.training_state.current_epoch = self.current_epoch
            self.training_state.global_step = self.state.global_step
            self.training_state.is_eval = True
            self.training_state.total_epochs = int(self.args.num_train_epochs)
            
            outputs = model(
                question_input_ids=inputs_on_device['question_input_ids'],
                question_attention_mask=inputs_on_device['question_attention_mask'],
                targets_input_ids=inputs_on_device['targets_input_ids'],
                targets_attention_mask=inputs_on_device.get('targets_attention_mask'),
                training_state=self.training_state,
                eval_mode=True,
            )
            
            loss = outputs.loss if has_labels else None
            
            # 
            if has_labels:
                if not hasattr(self, '_eval_loss_decode_sum'):
                    self._eval_loss_decode_sum = 0.0
                    self._eval_num_batches = 0
                
                if outputs.loss_reconstruct is not None:
                    self._eval_loss_decode_sum += outputs.loss_reconstruct.mean().item()
                self._eval_num_batches += 1
                
                if not hasattr(self, '_eval_num_samples'):
                    self._eval_num_samples = 0
                self._eval_num_samples += inputs_on_device['question_input_ids'].size(0)
            
            # 
            # DEBUG:  eval_acc 
            
            # 
            logits_is_none = outputs.logits is None
            has_labels_false = not has_labels
            ans_start_none = self.model.answer_start_id is None
            ans_end_none = self.model.answer_end_id is None
            
            if logits_is_none or has_labels_false or ans_start_none or ans_end_none:
                logger.warning(f"[EVAL WARNING] logits_is_none={logits_is_none}, has_labels={has_labels}, "
                              f"answer_start_id={self.model.answer_start_id}, answer_end_id={self.model.answer_end_id}")
            
            if outputs.logits is not None and has_labels:
                generated_answer_logits = outputs.logits
                generated_answer_ids = torch.argmax(generated_answer_logits, dim=-1)
                target_answer_ids = inputs_on_device['targets_input_ids'][-1]
                
                batch_size = generated_answer_ids.size(0)
                
                # 
                gen_ids = generated_answer_ids[0]
                tgt_ids = target_answer_ids[0]
                
                #  -100（label ignore mask）， tokenizer.decode 
                valid_gen_mask = gen_ids >= 0
                valid_tgt_mask = tgt_ids >= 0
                gen_ids_valid = gen_ids[valid_gen_mask]
                tgt_ids_valid = tgt_ids[valid_tgt_mask]
                
                # 
                full_gen_text = self.model.tokenizer.decode(gen_ids_valid, skip_special_tokens=False)
                full_tgt_text = self.model.tokenizer.decode(tgt_ids_valid, skip_special_tokens=False)
                
                logger.debug(f"[EVAL DEBUG] =====  0  =====")
                logger.debug(f"[EVAL DEBUG] : {full_gen_text}")
                logger.debug(f"[EVAL DEBUG] : {full_tgt_text}")
                
                # token
                start_pos_gen = (gen_ids == self.model.answer_start_id).nonzero(as_tuple=True)[0] if self.model.answer_start_id is not None else []
                end_pos_gen = (gen_ids == self.model.answer_end_id).nonzero(as_tuple=True)[0] if self.model.answer_end_id is not None else []
                start_pos_tgt = (tgt_ids == self.model.answer_start_id).nonzero(as_tuple=True)[0] if self.model.answer_start_id is not None else []
                end_pos_tgt = (tgt_ids == self.model.answer_end_id).nonzero(as_tuple=True)[0] if self.model.answer_end_id is not None else []
                
                logger.debug(f"[EVAL DEBUG]  answer_start/end : {start_pos_gen.tolist()}/{end_pos_gen.tolist() if len(end_pos_gen) > 0 else 'NOT FOUND'}")
                logger.debug(f"[EVAL DEBUG]  answer_start/end : {start_pos_tgt.tolist()}/{end_pos_tgt.tolist() if len(end_pos_tgt) > 0 else 'NOT FOUND'}")
                
                batch_correct = []
                
                for i in range(batch_size):
                    gen_ids = generated_answer_ids[i]
                    tgt_ids = target_answer_ids[i]
                    
                    # ANSWER_STARTANSWER_END
                    start_positions = (gen_ids == self.model.answer_start_id).nonzero(as_tuple=True)[0] if self.model.answer_start_id is not None else []
                    end_positions = (gen_ids == self.model.answer_end_id).nonzero(as_tuple=True)[0] if self.model.answer_end_id is not None else []
                    
                    if len(start_positions) > 0 and len(end_positions) > 0 and start_positions[0].item() < end_positions[0].item():
                        start_pos = start_positions[0].item()
                        end_pos = end_positions[0].item()
                        gen_answer_ids = gen_ids[start_pos:end_pos+1]
                        
                        tgt_start_positions = (tgt_ids == self.model.answer_start_id).nonzero(as_tuple=True)[0] if self.model.answer_start_id is not None else []
                        tgt_end_positions = (tgt_ids == self.model.answer_end_id).nonzero(as_tuple=True)[0] if self.model.answer_end_id is not None else []
                        
                        if len(tgt_start_positions) > 0 and len(tgt_end_positions) > 0:
                            tgt_start_pos = tgt_start_positions[0].item()
                            tgt_end_pos = tgt_end_positions[0].item()
                            tgt_answer_ids = tgt_ids[tgt_start_pos:tgt_end_pos+1]
                            
                            gen_text = self.model.tokenizer.decode(gen_answer_ids, skip_special_tokens=False)
                            tgt_text = self.model.tokenizer.decode(tgt_answer_ids, skip_special_tokens=False)
                            
                            gen_answer = extract_answer_from_text(gen_text)
                            tgt_answer = extract_answer_from_text(tgt_text)
                            
                            logger.debug(f"[EVAL DEBUG]  {i}: gen='{gen_answer}' vs tgt='{tgt_answer}' -> {'✓' if gen_answer == tgt_answer else '✗'}")
                            
                            is_correct = compare_answers(gen_answer, tgt_answer)
                            batch_correct.append(1 if is_correct else 0)
                        else:
                            logger.debug(f"[EVAL DEBUG]  {i}: token")
                            batch_correct.append(0)
                    else:
                        logger.debug(f"[EVAL DEBUG]  {i}: token")
                        batch_correct.append(0)
                
                logger.debug(f"[EVAL DEBUG] : {sum(batch_correct)}/{batch_size}")
                
                # （evaluation_loopeval_acc）
                if not hasattr(self, '_eval_correct_count'):
                    self._eval_correct_count = 0
                self._eval_correct_count += sum(batch_correct)
        
        logits = None
        labels = None
        
        return (loss, logits, labels)
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """，Backbone"""
        if self.freeze_backbone:
            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in self.model.named_parameters() 
                              if p.requires_grad and 'backbone' not in n],
                    'lr': self.args.learning_rate,
                }
            ]
        else:
            backbone_lr = self.backbone_learning_rate if self.backbone_learning_rate is not None else 1e-6
            main_lr = self.args.learning_rate
            
            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in self.model.named_parameters() 
                              if p.requires_grad and 'backbone' in n],
                    'lr': backbone_lr,
                },
                {
                    'params': [p for n, p in self.model.named_parameters() 
                              if p.requires_grad and 'backbone' not in n],
                    'lr': main_lr,
                }
            ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            weight_decay=self.args.weight_decay
        )
        
        self.lr_scheduler = self.create_scheduler(
            num_training_steps=num_training_steps,
            optimizer=self.optimizer
        )
    
    def _save(self, output_dir=None, state_dict=None):
        """
        ：，checkpoint
        
        ：
        1. LoRA（）
        2. PLaT（encoder, thinker, decode_proj）
        3. 
        4. tokenizer
        5. PLaT
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # （DDP）
        model = self.model
        if hasattr(model, 'module'):
            model = model.module
        
        # LoRA
        has_lora = hasattr(model.backbone, 'peft_config')
        
        if has_lora:
            # LoRA：adapter
            self._save_lora_checkpoint(model, output_dir)
        else:
            # LoRA：
            self._save_incremental_checkpoint(model, output_dir)
        
        # 
        try:
            import wandb
            wandb_available = wandb.run is not None
        except:
            wandb_available = False
        
        metadata = {
            'run_name': self.run_name,
            'wandb_run_id': wandb.run.id if wandb_available else None,
            'wandb_run_name': wandb.run.name if wandb_available else None,
            'current_epoch': getattr(self, 'current_epoch', 0),
            'global_step': self.state.global_step,
            'has_lora': has_lora,
            'base_model_path': self.plat_config.model.sft_model_path or self.plat_config.model.base_model_name,
        }
        metadata_path = os.path.join(output_dir, 'training_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # PLaT
        config_path = os.path.join(output_dir, 'plat_config.yaml')
        self.plat_config.save(config_path)
        
        # tokenizer
        if self.tokenizer_to_save is not None:
            self.tokenizer_to_save.save_pretrained(output_dir)
            logger.info(f"  tokenizercheckpoint: {output_dir}")
        
        # trainer state（）
        self.state.save_to_json(os.path.join(output_dir, 'trainer_state.json'))
        
        # checkpoint
        if output_dir and 'final' not in output_dir:
            self._cleanup_old_optimizer_states(output_dir)
        
        logger.info(f"  Checkpoint saved to: {output_dir}")
    
    def _save_lora_checkpoint(self, model, output_dir):
        """
        LoRA checkpoint
        
        ：
        1. LoRA adapter
        2. PLaT
        """
        # LoRA adapter
        lora_dir = os.path.join(output_dir, 'lora_adapter')
        
        #  config  model_type， AutoModel 
        if not hasattr(model.backbone.config, 'model_type') or model.backbone.config.model_type is None:
            sft_model_path = self.plat_config.model.sft_model_path or self.plat_config.model.base_model_name
            try:
                from transformers import AutoConfig
                base_config = AutoConfig.from_pretrained(sft_model_path)
                if hasattr(base_config, 'model_type') and base_config.model_type:
                    model.backbone.config.model_type = base_config.model_type
                    logger.info(f" model_type: {base_config.model_type}")
                else:
                    raise ValueError(f" {sft_model_path}  model_type")
            except Exception as e:
                logger.error(f" model_type : {e}")
                raise e
        
        model.backbone.save_pretrained(lora_dir)
        logger.info(f"  LoRA adapter: {lora_dir}")
        
        # PLaT
        self._save_plat_modules(model, output_dir)
    
    def _save_incremental_checkpoint(self, model, output_dir):
        """
        checkpoint（LoRA）
        
        ：
        1. 
        2. PLaT（backbone）
        """
        if self.freeze_backbone:
            # backbone：PLaT
            self._save_plat_modules(model, output_dir)
        else:
            # backbone：（）
            model_path = os.path.join(output_dir, 'pytorch_model.bin')
            torch.save(model.state_dict(), model_path)
            logger.info(f"  : {model_path}")
    
    def _save_plat_modules(self, model, output_dir):
        """
        PLaT
        
        ：
        - encoder: 
        - decode_proj: 
        - thinker（extra_layers）
        - thinker_head（head）
        - slot_embeddings（）
        """
        plat_state_dict = {}
        
        # PLaT
        plat_module_prefixes = [
            'encoder.',
            'decode_proj.',
            'thinker.',
            'thinker_head.',
            'slot_embeddings',
        ]
        
        for name, param in model.named_parameters():
            # backbone（PLaT）
            if 'backbone' in name:
                continue
            
            # PLaT
            for prefix in plat_module_prefixes:
                if name.startswith(prefix):
                    plat_state_dict[name] = param.data.clone()
                    break
            else:
                # backbone
                if 'backbone' not in name:
                    plat_state_dict[name] = param.data.clone()
        
        if plat_state_dict:
            plat_path = os.path.join(output_dir, 'plat_modules.bin')
            torch.save(plat_state_dict, plat_path)
            logger.info(f"  PLaT ({len(plat_state_dict)} ): {plat_path}")
    
    def _cleanup_old_optimizer_states(self, current_checkpoint_dir):
        """checkpoint"""
        if current_checkpoint_dir is None:
            return
        
        main_output_dir = self.args.output_dir
        
        checkpoint_dirs = []
        if os.path.isdir(main_output_dir):
            for item in os.listdir(main_output_dir):
                item_path = os.path.join(main_output_dir, item)
                if os.path.isdir(item_path) and item.startswith('checkpoint-'):
                    checkpoint_dirs.append(item_path)
        
        optimizer_files = [
            'optimizer.pt',
            'scheduler.pt', 
            'rng_state.pth',
            'scaler.pt'
        ]
        
        for ckpt_dir in checkpoint_dirs:
            if ckpt_dir != current_checkpoint_dir:
                for file_name in optimizer_files:
                    file_path = os.path.join(ckpt_dir, file_name)
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                            print(f"  : {file_path}")
                        except Exception as e:
                            print(f"  :  {file_path}: {e}")
