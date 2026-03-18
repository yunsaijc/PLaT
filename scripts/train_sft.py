# -*- coding: utf-8 -*-
"""
SFT 
 eval_acc ，。
"""
import os
import sys
import argparse
import logging
import shutil
import json
import torch
import numpy as np
from datetime import datetime

# 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from transformers import AutoTokenizer, Trainer, TrainingArguments, DefaultDataCollator
from plat.config import CoTSFTConfig, STEP_START, STEP_END, ANSWER_START, ANSWER_END
from plat.models.cot_model import CoTModel
from plat.data.reasoning_dataset import get_gsm8k_dataset
from plat.utils.helpers import compare_answers

logger = logging.getLogger(__name__)

def compute_metrics(eval_preds,
                    tokenizer,
                    output_dir,
                    global_step,
                    model,
                    questions,
                    answers,
                    gt_steps,
                    max_new_tokens,
                    eval_batch_size,
                    question_max_length):
    """
    （， entire steps + ），。
    """
    model.eval()
    sample_results = []
    correct_samples = 0

    with torch.no_grad():
        for i in range(0, len(questions), eval_batch_size):
            batch_questions = questions[i:i + eval_batch_size]
            batch_answers = answers[i:i + eval_batch_size]
            batch_steps = gt_steps[i:i + eval_batch_size]

            final_answers, reasoning_steps = model.inference(
                batch_questions,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                max_question_length=question_max_length
            )

            for question, gt_answer, gt_step_list, pred_answer, pred_full_text in zip(
                batch_questions, batch_answers, batch_steps, final_answers, reasoning_steps
            ):
                pred_answer = pred_answer.strip()
                is_correct = compare_answers(pred_answer, gt_answer)
                if is_correct:
                    correct_samples += 1

                #  gt_seq 
                gt_seq = "".join([f"{STEP_START}{step}{STEP_END}" for step in gt_step_list]) + f"{ANSWER_START}{gt_answer}{ANSWER_END}"
                pred_seq = pred_full_text if pred_full_text else ""

                sample_results.append({
                    "input": question,
                    "pred_seq": pred_seq,
                    "gt_seq": gt_seq,
                    "gt_answer": gt_answer,
                    "pred_answer": pred_answer,
                    "gt_steps": gt_step_list,
                    "pred_steps": pred_full_text,
                    "is_correct": bool(is_correct)
                })

    total_samples = len(questions)
    accuracy = correct_samples / total_samples if total_samples > 0 else 0.0

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        #  eval json ，
        old_files = [f for f in os.listdir(output_dir) if f.startswith("eval_answers_step_") and f.endswith(".json")]
        for old_file in old_files:
            try:
                os.remove(os.path.join(output_dir, old_file))
            except Exception as e:
                logger.warning(f": {e}")
        
        save_path = os.path.join(output_dir, f"eval_answers_step_{global_step}.json")
        log_data = {
            "metadata": {
                "step": global_step,
                "eval_acc": float(accuracy),
                "total_samples": total_samples,
                "correct_samples": correct_samples,
            },
            "samples": sample_results
        }
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(log_data, f, ensure_ascii=False, indent=4)
            logger.info(f"[Eval] Step {global_step} : {accuracy:.4f}, : {save_path}\n")
        except Exception as e:
            logger.error(f": {e}")

    return {"eval_acc": float(accuracy)}

def preprocess_logits_for_metrics(logits, labels):
    """
     logits， Token ID 
     labels 
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    
    # logits : (batch_size, seq_len, vocab_size)
    # : (batch_size, seq_len)
    preds = logits.argmax(dim=-1)
    
    #  labels 
    if labels is not None and preds.shape != labels.shape:
        # ，
        min_len = min(preds.shape[-1], labels.shape[-1])
        preds = preds[..., :min_len]
    
    return preds

class SFTTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.custom_tokenizer = kwargs.pop('tokenizer', None)
        super().__init__(*args, **kwargs)

    def _save(self, output_dir=None, state_dict=None):
        #  checkpoint  model_type  pad_token_id
        if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'config'):
            #  model_type
            if not hasattr(self.model.backbone.config, 'model_type') or self.model.backbone.config.model_type is None:
                try:
                    from transformers import AutoConfig
                    base_config = AutoConfig.from_pretrained(self.model.backbone.config._name_or_path)
                    if hasattr(base_config, 'model_type') and base_config.model_type:
                        self.model.backbone.config.model_type = base_config.model_type
                except:
                    pass
            #  pad_token_id
            if self.custom_tokenizer is not None:
                self.model.backbone.config.pad_token_id = self.custom_tokenizer.convert_tokens_to_ids('[PAD]')
        
        super()._save(output_dir, state_dict)
        if self.custom_tokenizer is not None:
            target_dir = output_dir or self.args.output_dir
            self.custom_tokenizer.save_pretrained(target_dir)
        self._cleanup_old_optimizer_states(output_dir)

    def _cleanup_old_optimizer_states(self, current_checkpoint_dir):
        if current_checkpoint_dir is None: return
        main_output_dir = self.args.output_dir
        if not os.path.isdir(main_output_dir): return
        checkpoint_dirs = [
            os.path.join(main_output_dir, d) for d in os.listdir(main_output_dir)
            if d.startswith('checkpoint-') and os.path.isdir(os.path.join(main_output_dir, d))
        ]
        files_to_delete = ['optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'scaler.pt']
        for ckpt_dir in checkpoint_dirs:
            if os.path.abspath(ckpt_dir) != os.path.abspath(current_checkpoint_dir):
                for f in files_to_delete:
                    f_path = os.path.join(ckpt_dir, f)
                    if os.path.exists(f_path):
                        try:
                            os.remove(f_path)
                        except:
                            pass

def main():
    parser = argparse.ArgumentParser(description="CoT SFT ")
    parser.add_argument("--config", type=str, required=True, help="YAML ")
    args = parser.parse_args()

    # 1. 
    config = CoTSFTConfig.load(args.config)

    # 
    prefix = config.wandb.experiment if hasattr(config, 'wandb') and config.wandb.enabled else "sft"
    run_name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(config.train.model_save_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    #  config.yaml （）
    config.save(os.path.join(output_dir, "config.yaml"))
    
    # （ checkpoint ）
    scripts_to_save = [
        '/home/jc/workspace/inf/plat/scripts/train_sft.py',
        '/home/jc/workspace/inf/plat/models/cot_model.py',
        '/home/jc/workspace/inf/plat/models/base_reasoning_model.py',
    ]
    for src_path in scripts_to_save:
        if os.path.exists(src_path):
            dst_path = os.path.join(output_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dst_path)
            logger.info(f" {os.path.basename(src_path)}  checkpoint")

    logger.info(f" config.yaml : {output_dir}")

    # 2.  Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    sft_tokens = [STEP_START, STEP_END, ANSWER_START, ANSWER_END]
    tokenizer.add_special_tokens({'additional_special_tokens': sft_tokens})
    
    # 3. 
    config.vocab_size = len(tokenizer)
    model = CoTModel.from_config(config)
    
    #  Token  Embedding
    old_vocab_size = model.backbone.config.vocab_size
    if len(tokenizer) > old_vocab_size:
        logger.info(f" Token  Embedding ( {old_vocab_size}  {len(tokenizer)})")
        embeddings = model.backbone.get_input_embeddings()
        
        #  PAD token（）
        if tokenizer.pad_token == '[PAD]':
            pad_id = tokenizer.pad_token_id
            eos_id = tokenizer.eos_token_id
            embeddings.weight.data[pad_id] = embeddings.weight.data[eos_id].clone()
            logger.info(f" '[PAD]' (ID: {pad_id}) ")
        
        #  SFT  Token（ '.'  embedding ）
        source_token_id = tokenizer.convert_tokens_to_ids('<')
        for token in sft_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id != tokenizer.unk_token_id and token_id >= old_vocab_size:
                embeddings.weight.data[token_id] = embeddings.weight.data[source_token_id].clone()
                logger.info(f" '{token}' (ID: {token_id}) ")
    
    model.set_tokenizer(tokenizer)

    #  backbone config  model_type（ checkpoint）
    if not hasattr(model.backbone.config, 'model_type') or model.backbone.config.model_type is None:
        try:
            from transformers import AutoConfig
            base_config = AutoConfig.from_pretrained(config.base_model_name)
            if hasattr(base_config, 'model_type') and base_config.model_type:
                model.backbone.config.model_type = base_config.model_type
                logger.info(f" model_type: {base_config.model_type}")
            else:
                raise ValueError(f" {config.base_model_name}  model_type")
        except Exception as e:
            logger.error(f" model_type : {e}")
            raise e

    # 4. 
    cache_dir = os.path.join(config.data.data_dir, "cache")
    dataset = get_gsm8k_dataset(data_dir=config.data.data_dir, cache_dir=cache_dir)

    eval_questions = list(dataset["valid"]["question"])
    eval_answers = list(dataset["valid"]["answer"])
    eval_steps = list(dataset["valid"]["steps"])

    def tokenize_fn(ex):
        # batched=True ，ex ，
        batch_size = len(ex['question'])
        results = {
            'input_ids': [],
            'attention_mask': [],
            'labels': []
        }
        
        step_start_id = tokenizer.convert_tokens_to_ids(STEP_START)
        answer_start_id = tokenizer.convert_tokens_to_ids(ANSWER_START)
        
        for i in range(batch_size):
            # 
            full_text = ex['question'][i] + \
                       "".join([f"{STEP_START}{p}{STEP_END}" for p in ex['steps'][i]]) + \
                       f"{ANSWER_START}{ex['answer'][i]}{ANSWER_END}"
            
            # Tokenize
            res = tokenizer(full_text, truncation=True, padding='max_length', max_length=config.data.max_length)
            labels = list(res["input_ids"])
            
            #  labels： STEP_START  ANSWER_START  -100
            if step_start_id in labels:
                idx = labels.index(step_start_id)
                for j in range(idx): 
                    labels[j] = -100
            elif answer_start_id in labels:
                idx = labels.index(answer_start_id)
                for j in range(idx):
                    labels[j] = -100
            
            results['input_ids'].append(res["input_ids"])
            results['attention_mask'].append(res["attention_mask"])
            results['labels'].append(labels)
        
        return results

    tokenized_ds = dataset.map(
        tokenize_fn, 
        remove_columns=dataset['train'].column_names,
        num_proc=config.train.preprocessing_num_workers,
        batched=True
    )

    # 5.  ()
    if config.wandb.enabled:
        os.environ["WANDB_PROJECT"] = config.wandb.project
        os.environ["WANDB_RUN_GROUP"] = config.wandb.experiment
    
    train_args = TrainingArguments(
        output_dir=output_dir,  #  output_dir， datetime.now()
        per_device_train_batch_size=config.train.per_device_train_batch_size,
        per_device_eval_batch_size=config.train.per_device_eval_batch_size,
        learning_rate=config.train.learning_rate,
        num_train_epochs=config.train.num_train_epochs,
        logging_steps=config.train.logging_steps,
        save_steps=config.train.save_steps,
        eval_strategy="steps",
        eval_steps=config.train.eval_steps,
        lr_scheduler_type=config.train.lr_scheduler_type,
        warmup_ratio=config.train.warmup_ratio,
        bf16=config.train.bf16,
        fp16=config.train.fp16,
        report_to="wandb" if config.wandb.enabled else "none",
        run_name=run_name,
        seed=config.train.seed,
        save_total_limit=config.train.save_total_limit,
        # ： eval_acc 
        load_best_model_at_end=True,
        metric_for_best_model="eval_acc",
        greater_is_better=True,
        dataloader_num_workers=config.train.dataloader_num_workers,
        include_inputs_for_metrics=True,
    )

    # 6. 
    trainer = None
    def wrapped_compute_metrics(eval_preds):
        current_step = trainer.state.global_step if hasattr(trainer, 'state') else 0
        return compute_metrics(
            eval_preds,
            tokenizer,
            train_args.output_dir,
            current_step,
            trainer.model,
            eval_questions,
            eval_answers,
            eval_steps,
            config.data.max_length,
            train_args.per_device_eval_batch_size,
            config.data.max_length
        )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["valid"],
        data_collator=DefaultDataCollator(),
        tokenizer=tokenizer,
        compute_metrics=wrapped_compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    logger.info("---  SFT  ---")
    trainer.train()
    
    # 7. 
    final_path = os.path.join(train_args.output_dir, "final_model")
    
    #  config  model_type， AutoModel 
    #  base_model_name  model_type
    if not hasattr(model.backbone.config, 'model_type') or model.backbone.config.model_type is None:
        try:
            from transformers import AutoConfig
            base_config = AutoConfig.from_pretrained(config.base_model_name)
            if hasattr(base_config, 'model_type') and base_config.model_type:
                model.backbone.config.model_type = base_config.model_type
                logger.info(f" model_type: {base_config.model_type}")
            else:
                raise ValueError(f" {config.base_model_name}  model_type")
        except Exception as e:
            logger.error(f" model_type : {e}")
            raise e

    #  pad_token_id， transformers  eos_token_id 
    model.backbone.config.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    model.backbone.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    config.save(os.path.join(final_path, "config.yaml"))
    logger.info(f"，: {final_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
