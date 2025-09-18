import os
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig, ListConfig
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed,
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
        self.model_args = cfg.model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            use_fast=self.model_args.use_fast_tokenizer,
            trust_remote_code=True,
            cache_dir=self.model_args.cache_dir,
            token=cfg.token,
            revision=self.model_args.revision,
            padding_side=self.model_args.padding_side,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.raw_datasets = DatasetDict()
        
        if self.data_args.train_file:
            train_data = load_dataset(os.path.join(parent_dir, self.data_args.train_file))
            if train_data:
                self.raw_datasets["train"] = Dataset.from_list(train_data)
        if self.data_args.validation_file:
            val_data = load_dataset(os.path.join(parent_dir, self.data_args.validation_file))
            if val_data:
                self.raw_datasets["validation"] = Dataset.from_list(val_data)
        if self.data_args.test_file:
            test_data = load_dataset(os.path.join(parent_dir, self.data_args.test_file))
            if test_data:
                self.raw_datasets["test"] = Dataset.from_list(test_data)

    def format_prompt(self, text: str, label: str = None) -> str:
        """Format prompt for training with proper completion"""
        if label is not None:
            prompt_str = PROMPT_TEMPLATE.format(text=text)
            messages = [
                {"role": "user", "content": prompt_str},
                {"role": "assistant", "content": label}
            ]
            
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False
            )
        else:
            prompt_str = PROMPT_TEMPLATE.format(text=text)
            messages = [
                {"role": "user", "content": prompt_str}
            ]
            
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

    def preprocess_fn(self, examples):
        processed_texts = []
        for text, label in zip(examples[self.data_args.text_col], examples[self.data_args.label_col]):
            formatted_text = self.format_prompt(text, label)
            processed_texts.append(formatted_text)
        return {"text": processed_texts}

    def get_processed_datasets(self, training_args) -> DatasetDict:
        remain_columns = ["text"]
        
        if "train" in self.raw_datasets:
            column_names = self.raw_datasets["train"].column_names
        elif "validation" in self.raw_datasets:
            column_names = self.raw_datasets["validation"].column_names
        elif "test" in self.raw_datasets:
            column_names = self.raw_datasets["test"].column_names
        else:
            raise ValueError("No datasets available for processing")
        
        with training_args.main_process_first(desc="dataset mapping"):
            datasets_tokenized = self.raw_datasets.map(
                self.preprocess_fn,
                batched=True,
                remove_columns=list(set(column_names) - set(remain_columns)),
                load_from_cache_file=not self.data_args.overwrite_cache,
                num_proc=self.data_args.preprocessing_num_workers,
                desc="Format dataset with prompts",
            )

        if training_args.do_train:
            if "train" not in datasets_tokenized:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = datasets_tokenized["train"]
            if self.data_args.shuffle:
                train_dataset = train_dataset.shuffle(seed=training_args.seed)
            if self.data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), self.data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))
            datasets_tokenized["train"] = train_dataset

        if training_args.do_eval:
            if "validation" not in datasets_tokenized:
                if "test" not in datasets_tokenized:
                    raise ValueError("--do_eval requires a validation dataset")
                else:
                    logger.warning("--do_eval requires a validation dataset, using test dataset instead")
                    eval_dataset = datasets_tokenized["test"]
            else:
                eval_dataset = datasets_tokenized["validation"]

            if self.data_args.max_eval_samples is not None:
                max_eval_samples = min(len(eval_dataset), self.data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
            datasets_tokenized["validation"] = eval_dataset

        if "test" in datasets_tokenized and self.data_args.max_test_samples is not None:
            max_test_samples = min(len(datasets_tokenized["test"]), self.data_args.max_test_samples)
            datasets_tokenized["test"] = datasets_tokenized["test"].select(range(max_test_samples))

        return datasets_tokenized


class LLMFinetuning:
    def __init__(self, cfg: DictConfig) -> None:
        self.data_args = cfg.dataset
        self.model_args = cfg.model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            use_fast=self.model_args.use_fast_tokenizer,
            trust_remote_code=True,
            cache_dir=self.model_args.cache_dir,
            token=cfg.token,
            revision=self.model_args.revision,
            padding_side=self.model_args.padding_side,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.verbose = cfg.logging.verbose
        
        logger.info(f"Lora: {self.model_args.lora}, qLora: {self.model_args.qlora}")

        bnb_config = None
        self.compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if hasattr(self.model_args, 'qlora') and self.model_args.qlora:
            logger.info("Using QLoRA with 4-bit quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.compute_dtype
            )
        else:
            logger.info("Using full precision training (no quantization)")
        
        config = AutoConfig.from_pretrained(
            self.model_args.model_name_or_path,
            pad_token_id=self.tokenizer.pad_token_id,
            trust_remote_code=self.model_args.trust_remote_code,
            cache_dir=self.model_args.cache_dir,
            use_cache=self.model_args.use_cache,
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
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation=self.model_args.attn_implementation,
            torch_dtype=self.compute_dtype,
            low_cpu_mem_usage=self.model_args.low_cpu_mem_usage,
        )

        if self.model_args.lora is not None and self.model_args.lora != 'None':
            modules = find_all_linear_names(self.model)
            if self.model_args.qlora:
                self.model = prepare_model_for_kbit_training(self.model)
            modules_to_save = self.model_args.lora.modules_to_save
            if isinstance(modules_to_save, ListConfig):
                modules_to_save = list(modules_to_save)
        
            lora_config = LoraConfig(
                r=self.model_args.lora.r,
                lora_alpha=self.model_args.lora.lora_alpha,
                target_modules=modules,
                lora_dropout=self.model_args.lora.lora_dropout,
                bias=self.model_args.lora.bias,
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lora_config)
        
            train_p, tot_p = self.model.get_nb_trainable_parameters()
            logger.info(f"Model loaded: {self.model_args.model_name_or_path}")
            logger.info(f'Trainable parameters:      {train_p/1e6:.2f}M')
            logger.info(f'Total parameters:          {tot_p/1e6:.2f}M')
            logger.info(f'% of trainable parameters: {100*train_p/tot_p:.2f}%')
            
        self.metrics = EvaluateMetrics()
    
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
            compute_metrics=compute_metrics,
            callbacks=[MemoryLoggerCallback(), TimeLoggerCallback()],
        )
        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        if hasattr(self.model, 'save_pretrained'):
            logger.info("Saving LoRA adapters...")
            self.model.save_pretrained(training_args.output_dir)
        
        logger.info(f"Model saved to {training_args.output_dir}")
                
        return trainer
    
    def evaluate(self):
        test_data_path = os.path.join(parent_dir, self.data_args.test_file)
        test_data = load_dataset(test_data_path)
        
        predictions = []
        references = []
        inputs = []
        
        for item in test_data:
            input_text = item['text']
            reference = item['label']
            
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
                    'lexical_similarity': result.lexical_similarity,
                    'num_samples': result.num_samples
                },
                'detailed_results': detailed_results
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Evaluation results saved to {results_file}")
        
        return result
    
    def predict(self, text: str) -> str:
        self.model.eval()
        
        if self.verbose:
            if hasattr(self.model, 'peft_config'):
                logger.debug("✅ LoRA adapters are active")
            else:
                logger.warning("⚠️ LoRA adapters might not be active")
        
        prompt_str = PROMPT_TEMPLATE.format(text=text)
        messages = [
            {"role": "user", "content": prompt_str},
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        result = self._generate_text(
            prompt,
            max_new_tokens=256,
            temperature=0.3,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
        )
        
        logger.debug(f"Raw prediction: '{result}'")
        
        result = result.strip()
        
        if result.startswith("### Địa chỉ hoàn thiện:"):
            result = result.replace("### Địa chỉ hoàn thiện:", "").strip()
        
        logger.debug(f"Cleaned prediction: '{result}'")
        
        return result

    def _generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.05,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=self.compute_dtype):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                )
        
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        
        if self.verbose:
            logger.debug(f"Generated tokens shape: {generated_tokens.shape}")
            logger.debug(f"Generated token IDs: {generated_tokens[:10]}")
        
        if len(generated_tokens) == 1:
            eos_token_id = self.tokenizer.eos_token_id
            if generated_tokens[0].item() == eos_token_id:
                logger.warning(f"⚠️ Model only generated EOS token ({eos_token_id}). This suggests the model hasn't learned to generate proper responses.")
        
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text.strip()
    

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
        trainer = LLMFinetuning(cfg=cfg)
        training_args = SFTConfig(**cfg.training_arguments)
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        
        training_args.dataloader_pin_memory = False
        training_args.remove_unused_columns = True
        training_args.dataloader_num_workers = 0
        training_args.fp16 = not torch.cuda.is_bf16_supported()
        training_args.bf16 = torch.cuda.is_bf16_supported()
        
        logger.info(f"Mixed precision: BF16={training_args.bf16}, FP16={training_args.fp16}, TF32={training_args.tf32}")
        
        datasets_processed = dataloader.get_processed_datasets(training_args)
        train_dataset = datasets_processed.get("train")
        val_dataset = datasets_processed.get("validation")
        test_dataset = datasets_processed.get("test")
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"Validation dataset size: {len(val_dataset)}")
        if test_dataset:
            logger.info(f"Test dataset size: {len(test_dataset)}")
        
        trainer.train(
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            training_args=training_args
        )
        
        if test_dataset:
            logger.info("Evaluating model on test dataset...")
            trainer.evaluate()


if __name__ == "__main__":
    main()
