import os
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

import math
import hydra
import torch
from loguru import logger
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig, ListConfig
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from src.callbacks.time_callback import TimeLoggerCallback
from src.callbacks.memory_callback import MemoryLoggerCallback
from src.utils import find_all_linear_names, load_dataset

load_dotenv()


class CPTDataloader:
    """Dataloader for Continued Pre-training.
    
    Loads pre-chunked JSON data (produced by prepare_cpt_data.py)
    with format [{"text": "chunk content"}, ...].
    No prompt formatting - uses raw text for language modeling.
    """

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
            train_data = load_dataset(os.path.join(project_root, self.data_args.train_file))
            if train_data:
                self.raw_datasets["train"] = Dataset.from_list(train_data)
        if self.data_args.validation_file:
            val_data = load_dataset(os.path.join(project_root, self.data_args.validation_file))
            if val_data:
                self.raw_datasets["validation"] = Dataset.from_list(val_data)

    def get_processed_datasets(self, training_args) -> DatasetDict:
        """Return datasets ready for training. No special processing needed
        for CPT since data is already chunked text."""
        datasets = self.raw_datasets

        if training_args.do_train:
            if "train" not in datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = datasets["train"]
            if self.data_args.shuffle:
                train_dataset = train_dataset.shuffle(seed=training_args.seed)
            if self.data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), self.data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))
            datasets["train"] = train_dataset

        if training_args.do_eval:
            if "validation" not in datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = datasets["validation"]
            if self.data_args.max_eval_samples is not None:
                max_eval_samples = min(len(eval_dataset), self.data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
            datasets["validation"] = eval_dataset

        return datasets


class CPTFinetuning:
    """Continued Pre-training (CPT) for domain adaptation.
    
    Trains the model on raw text using causal language modeling objective.
    Evaluates using perplexity (lower is better).
    """

    def __init__(self, cfg: DictConfig, tokenizer=None) -> None:
        self.cfg = cfg
        self.data_args = cfg.dataset
        self.model_args = cfg.model
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
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
                use_dora=self.model_args.lora.use_dora,
            )
            self.model = get_peft_model(self.model, lora_config)

            train_p, tot_p = self.model.get_nb_trainable_parameters()
            logger.info(f"Model loaded: {self.model_args.model_name_or_path}")
            logger.info(f'Trainable parameters:      {train_p/1e6:.2f}M')
            logger.info(f'Total parameters:          {tot_p/1e6:.2f}M')
            logger.info(f'% of trainable parameters: {100*train_p/tot_p:.2f}%')

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset = None,
        training_args=None,
    ):
        """Train the model using CPT (no compute_metrics needed, 
        eval_loss/perplexity is tracked automatically)."""
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[MemoryLoggerCallback(), TimeLoggerCallback()],
        )
        trainer.train()
        trainer.save_model()
        logger.info(f"Model saved to {training_args.output_dir}")

        if eval_dataset is not None:
            eval_result = trainer.evaluate()
            eval_loss = eval_result.get("eval_loss", None)
            if eval_loss is not None:
                perplexity = math.exp(eval_loss)
                logger.info(f"Final eval_loss: {eval_loss:.4f}")
                logger.info(f"Final perplexity: {perplexity:.4f}")

        return trainer


@hydra.main(version_base=None, config_path="../../conf", config_name="cpt-conf")
def main(cfg: DictConfig):
    set_seed(cfg.seed, deterministic=True)

    logger.info("Starting CPT Training")
    dataloader = CPTDataloader(cfg)
    trainer = CPTFinetuning(cfg=cfg, tokenizer=dataloader.tokenizer)
    training_args = SFTConfig(**cfg.training_arguments)
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    training_args.dataloader_pin_memory = False
    training_args.remove_unused_columns = True
    training_args.dataloader_num_workers = 0
    training_args.fp16 = not torch.cuda.is_bf16_supported()
    training_args.bf16 = torch.cuda.is_bf16_supported()

    logger.info(f"Mixed precision: BF16={training_args.bf16}, FP16={training_args.fp16}")

    datasets_processed = dataloader.get_processed_datasets(training_args)
    train_dataset = datasets_processed.get("train")
    val_dataset = datasets_processed.get("validation")

    logger.info(f"Train dataset size: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Validation dataset size: {len(val_dataset)}")

    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        training_args=training_args,
    )

    # Evaluate
    if val_dataset:
        from src.finetune.cpt.cpt_evaluation import CPTEvaluator
        evaluator = CPTEvaluator(cfg=cfg)
        evaluator.full_evaluation()


if __name__ == "__main__":
    main()
