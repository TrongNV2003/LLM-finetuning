import os
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import json
import hydra
import torch
import mlflow
from loguru import logger
from datasets import Dataset
from dotenv import load_dotenv
from omegaconf import DictConfig, ListConfig
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from src.callbacks.time_callback import TimeLoggerCallback
from src.callbacks.memory_callback import MemoryLoggerCallback
from src.utils import find_all_linear_names, load_dataset
from src.metrics import EvaluateMetrics, compute_metrics_fn
from src.prompts import PROMPT_TEMPLATE

load_dotenv()


class Dataloader:
    def __init__(self, cfg: DictConfig):
        self.data_args = cfg.dataset
        self.prompt_template = PROMPT_TEMPLATE
        self.train_data = load_dataset(os.path.join(parent_dir, self.data_args.train_file))
        self.val_data = load_dataset(os.path.join(parent_dir, self.data_args.validation_file))
        self.test_data = load_dataset(os.path.join(parent_dir, self.data_args.test_file))

    def format_prompt(self, text: str, label: str = None) -> str:
        if label:
            return self.prompt_template.format(text=text, label=label)
        else:
            return self.prompt_template.format(text=text, label="")

    def get_dataset(self, split: str = "train") -> Dataset:
        """Convert to HuggingFace Dataset format"""
        if split == "train" and self.train_data:
            data = self.train_data
        elif split == "validation" and self.val_data:
            data = self.val_data
        elif split == "test" and self.test_data:
            data = self.test_data
        else:
            logger.error(f"No data available for split: {split}")
            return None
            
        formatted_data = []
        
        for item in data:
            formatted_text = self.format_prompt(item[self.data_args.text_col], item[self.data_args.label_col])
            formatted_data.append({"text": formatted_text})
        
        return Dataset.from_list(formatted_data)

class LLMFinetuning:
    def __init__(self, cfg: DictConfig) -> None:
        self.prompt_template = PROMPT_TEMPLATE
        self.model_args = cfg.model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name_or_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype
        )
        
        config = AutoConfig.from_pretrained(
            self.model_args.model_name_or_path,
            pad_token_id=self.tokenizer.pad_token_id,
            trust_remote_code=self.model_args.trust_remote_code,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.revision,
            token=cfg.token,
        )
        config.deterministic_flash_attention = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name_or_path,
            trust_remote_code=self.model_args.trust_remote_code,
            config=config,
            token=cfg.token,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.revision,
            quantization_config=self.bnb_config,
            device_map="auto",
            attn_implementation=self.model_args.attn_implementation,
        )

        if self.model_args.lora is not None and self.model_args.lora != 'None':
            modules = find_all_linear_names(self.model)
            if self.model_args.qlora:
                self.model = prepare_model_for_kbit_training(self.model)
            modules_to_save = self.model_args.lora.modules_to_save
            if isinstance(modules_to_save, ListConfig):
                modules_to_save = list(modules_to_save)
        
        self.lora_config = LoraConfig(
            r=self.model_args.lora.r,
            lora_alpha=self.model_args.lora.lora_alpha,
            target_modules=modules,
            lora_dropout=self.model_args.lora.lora_dropout,
            bias=self.model_args.lora.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, self.lora_config)
        
        self.metrics = EvaluateMetrics()

        train_p, tot_p = self.model.get_nb_trainable_parameters()
        logger.info(f"Model loaded: {self.model_args.model_name_or_path}")
        logger.warning(f'Trainable parameters:      {train_p/1e6:.2f}M')
        logger.warning(f'Total parameters:          {tot_p/1e6:.2f}M')
        logger.warning(f'% of trainable parameters: {100*train_p/tot_p:.2f}%')
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset = None,
        training_args: TrainingArguments = None,
        **kwargs
    ):
        compute_metrics = None
        if eval_dataset is not None:
            compute_metrics = compute_metrics_fn(self.tokenizer)

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[MemoryLoggerCallback(), TimeLoggerCallback()],
            compute_metrics=compute_metrics,
        )
        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")
        
        return trainer
    
    def evaluate(self, test_dataset: Dataset):
        test_data = []
        for item in test_dataset:
            text = item['text']
            if "### Địa chỉ hoàn thiện:" in text:
                parts = text.split("### Địa chỉ hoàn thiện:")
                if len(parts) == 2:
                    input_section = parts[0]
                    if "### Địa chỉ gốc:" in input_section:
                        input_part = input_section.split("### Địa chỉ gốc:")[-1].strip()
                    else:
                        input_part = input_section.strip()
                    
                    reference = parts[1].strip()
                    test_data.append({
                        'input': input_part,
                        'reference': reference
                    })
        
        predictions = []
        references = []
        inputs = []
        
        for item in test_data:
            input_text = item['input']
            reference = item['reference']
            
            prediction = self.predict(input_text)
            
            predictions.append(prediction)
            references.append(reference)
            inputs.append(input_text)

        result = self.metrics.metrics_evaluate(predictions, references)
        self.metrics.print_evaluation_report(result, "Final Model Evaluation")

        detailed_results = []
        for i, (inp, pred, ref) in enumerate(zip(inputs, predictions, references)):
            detailed_results.append({
                'id': i,
                'input': inp,
                'prediction': pred,
                'reference': ref,
                'exact_match': pred.strip().lower() == ref.strip().lower()
            })
        
        results_file = "evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metrics': {
                    'exact_match': result.exact_match,
                    'bleu_score': result.bleu_score,
                    'rouge_l': result.rouge_l,
                    'semantic_similarity': result.semantic_similarity,
                    'bert_score': result.bert_score,
                    'num_samples': result.num_samples
                },
                'detailed_results': detailed_results
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Evaluation results saved to {results_file}")
        
        return result
    
    def predict(self, text: str) -> str:
        prompt = self.prompt_template.format(text=text, label="")
        result = self._generate_text(
            prompt,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True
        )
        return result.strip()

    def _generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> str:
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        prompt_length = len(prompt)
        generated_part = generated_text[prompt_length:].strip()
        
        return generated_part
    

@hydra.main(version_base=None, config_path="conf", config_name="finetuning")
def main(cfg: DictConfig):
    set_seed(cfg.seed, deterministic=True)

    experiment_name = cfg.logging.mlflow.experiment_name
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.warning(f"MLflow experiment creation failed: {e}")
        experiment_id = None
        
    with mlflow.start_run(run_name="finetuning"):
        dataloader = Dataloader(cfg)
        
        train_dataset = dataloader.get_dataset("train")
        val_dataset = dataloader.get_dataset("validation")
        test_dataset = dataloader.get_dataset("test")
        
        if train_dataset is None:
            logger.error("No training data available.")
            return
        
        trainer = LLMFinetuning(cfg=cfg)

        training_args = SFTConfig(**cfg.training_arguments)
        
        trainer.train(
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            training_args=training_args
        )
        
        if test_dataset:
            logger.info("Evaluating model on test dataset...")
            evaluation_result = trainer.evaluate(
                test_dataset, 
                output_dir=os.path.join(training_args.output_dir, "evaluation")
            )
            
            if evaluation_result:
                mlflow.log_metrics({
                    "exact_match": evaluation_result.exact_match,
                    "bleu_score": evaluation_result.bleu_score,
                    "rouge_l": evaluation_result.rouge_l,
                    "semantic_similarity": evaluation_result.semantic_similarity,
                    "bert_score": evaluation_result.bert_score,
                })
        
    
    # Test on sample addresses
        logger.info("Testing on sample addresses...")
        test_addresses = [
            "68, Xthuy, Cgiay, HN",
            "123, Ngtroi, Q1, HCM",
            "45, Lthanh, Hkiem, HN"
        ]
        
        for address in test_addresses:
            result = trainer.predict(address)
            logger.info(f"Input: {address}")
            logger.info(f"Output: {result}")
            logger.info("-" * 50)

if __name__ == "__main__":
    main()
